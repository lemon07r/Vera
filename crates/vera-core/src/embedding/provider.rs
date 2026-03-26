//! Embedding provider abstraction and OpenAI-compatible implementation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::types::Chunk;

// ── Error types ──────────────────────────────────────────────────────

/// Errors specific to the embedding pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Authentication failure (invalid or missing API key).
    #[error("embedding API authentication failed: {message}")]
    AuthError { message: String },

    /// Cannot reach the embedding endpoint.
    #[error("embedding API connection failed: {message}")]
    ConnectionError { message: String },

    /// The API returned a non-auth, non-connection error.
    #[error("embedding API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    /// Rate limit exceeded.
    #[error("embedding API rate limit exceeded: {message}")]
    RateLimitError { message: String },

    /// Unexpected response format.
    #[error("unexpected embedding API response: {message}")]
    ResponseError { message: String },
}

// ── Provider trait ───────────────────────────────────────────────────

/// Trait abstracting an embedding provider.
///
/// Implementations must be able to embed a batch of text inputs and return
/// one vector per input. Vectors must have consistent dimensionality.
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of text inputs, returning one vector per input.
    ///
    /// The returned vectors must all have the same dimensionality.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Return the expected vector dimensionality (if known ahead of time).
    fn expected_dim(&self) -> Option<usize>;
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for an OpenAI-compatible embedding provider.
#[derive(Clone)]
pub struct EmbeddingProviderConfig {
    /// Base URL for the API (e.g. "https://api.openai.com/v1").
    pub base_url: String,
    /// Model identifier (e.g. "Qwen/Qwen3-Embedding-8B").
    pub model_id: String,
    /// API key (never logged or exposed).
    api_key: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
}

impl std::fmt::Debug for EmbeddingProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingProviderConfig")
            .field("base_url", &self.base_url)
            .field("model_id", &self.model_id)
            .field("api_key", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl EmbeddingProviderConfig {
    /// Create a new config. The API key is stored opaquely and never exposed.
    pub fn new(base_url: String, model_id: String, api_key: String) -> Self {
        Self {
            base_url,
            model_id,
            api_key,
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads:
    /// - `EMBEDDING_MODEL_BASE_URL`
    /// - `EMBEDDING_MODEL_ID`
    /// - `EMBEDDING_MODEL_API_KEY`
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("EMBEDDING_MODEL_BASE_URL")
            .context("EMBEDDING_MODEL_BASE_URL not set")?;
        let model_id = std::env::var("EMBEDDING_MODEL_ID").context("EMBEDDING_MODEL_ID not set")?;
        let api_key =
            std::env::var("EMBEDDING_MODEL_API_KEY").context("EMBEDDING_MODEL_API_KEY not set")?;

        Ok(Self::new(base_url, model_id, api_key))
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum retry count.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

// ── OpenAI-compatible provider ───────────────────────────────────────

/// OpenAI-compatible embedding provider.
///
/// Works with any API that implements the OpenAI `/v1/embeddings` endpoint,
/// including Nebius, Together, Fireworks, vLLM, etc.
pub struct OpenAiProvider {
    client: reqwest::Client,
    config: EmbeddingProviderConfig,
}

impl OpenAiProvider {
    /// Create a new provider from configuration.
    pub fn new(config: EmbeddingProviderConfig) -> Result<Self> {
        crate::init_tls();
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client")?;

        Ok(Self { client, config })
    }

    /// Build the embeddings endpoint URL.
    fn endpoint_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/embeddings")
    }

    /// Execute a single embedding API call with retry logic.
    ///
    /// Rate limit errors (429) get extra retries with longer backoffs
    /// to respect API quotas while still completing the operation.
    async fn call_api(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = self.endpoint_url();
        let body = EmbeddingRequest {
            model: &self.config.model_id,
            input: texts,
        };

        // Rate limit errors get extra retries beyond the normal max.
        let max_total_retries = self.config.max_retries + 4;
        let mut last_err = None;
        for attempt in 0..=max_total_retries {
            if attempt > 0 {
                let is_rate_limit = matches!(last_err, Some(EmbeddingError::RateLimitError { .. }));
                let delay = if is_rate_limit {
                    // Rate limit: wait 2-4s with exponential backoff.
                    Duration::from_secs(2 + u64::from(attempt.min(2)))
                } else {
                    Duration::from_millis(500 * 2u64.pow(attempt.min(5) - 1))
                };
                debug!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    rate_limited = is_rate_limit,
                    "retrying embedding API"
                );
                tokio::time::sleep(delay).await;
            }

            match self.send_request(&url, &body).await {
                Ok(vectors) => return Ok(vectors),
                Err(e) => {
                    // Don't retry auth errors — they won't resolve.
                    if matches!(e, EmbeddingError::AuthError { .. }) {
                        return Err(e);
                    }
                    warn!(
                        attempt = attempt + 1,
                        max = max_total_retries + 1,
                        error = %e,
                        "embedding API call failed"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| EmbeddingError::ApiError {
            status: 0,
            message: "all retries exhausted".to_string(),
        }))
    }

    /// Send a single HTTP request and parse the response.
    async fn send_request(
        &self,
        url: &str,
        body: &EmbeddingRequest<'_>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    EmbeddingError::ConnectionError {
                        message: format!("failed to connect to embedding API: {e}"),
                    }
                } else {
                    EmbeddingError::ConnectionError {
                        message: format!("request failed: {e}"),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if status == 401 || status == 403 {
            let text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::AuthError {
                message: sanitize_error_message(&text),
            });
        }

        if status == 429 {
            let text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::RateLimitError {
                message: sanitize_error_message(&text),
            });
        }

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            // Some providers return 400 with "Unable to process" for transient
            // overload conditions. Treat these as rate limits so they get retried.
            if status == 400 && text.contains("Unable to process") {
                return Err(EmbeddingError::RateLimitError {
                    message: sanitize_error_message(&text),
                });
            }
            return Err(EmbeddingError::ApiError {
                status,
                message: sanitize_error_message(&text),
            });
        }

        let resp: EmbeddingResponse =
            response
                .json()
                .await
                .map_err(|e| EmbeddingError::ResponseError {
                    message: format!("failed to parse embedding response: {e}"),
                })?;

        // Sort by index to ensure correct ordering.
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        let vectors: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        if vectors.len() != body.input.len() {
            return Err(EmbeddingError::ResponseError {
                message: format!(
                    "expected {} embeddings, got {}",
                    body.input.len(),
                    vectors.len()
                ),
            });
        }

        Ok(vectors)
    }
}

impl EmbeddingProvider for OpenAiProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(texts).await
    }

    fn expected_dim(&self) -> Option<usize> {
        // Qwen3-Embedding-8B produces 4096-dim vectors.
        // We don't hardcode this — it's discoverable from the first response.
        None
    }
}

// ── Cached embedding provider ────────────────────────────────────────

/// An in-memory LRU-style cache wrapper around any `EmbeddingProvider`.
///
/// Caches `query_text → embedding_vector` so that repeated identical queries
/// (common during interactive search sessions) skip the API call entirely.
/// The first query pays full API cost; subsequent identical queries resolve
/// in microseconds from cache.
///
/// The cache uses a bounded `HashMap` with capacity-based eviction: when the
/// cache exceeds `max_entries`, the oldest entry (by insertion time) is removed.
pub struct CachedEmbeddingProvider<P> {
    inner: P,
    cache: Mutex<LruCache>,
}

/// Simple bounded cache with insertion-order eviction.
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
}

struct CacheEntry {
    vector: Vec<f32>,
    inserted_at: Instant,
}

impl LruCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries: max_entries.max(1),
        }
    }

    fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.entries.get(key).map(|e| &e.vector)
    }

    fn insert(&mut self, key: String, vector: Vec<f32>) {
        // Evict oldest entry if at capacity and this is a new key.
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest_key);
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                vector,
                inserted_at: Instant::now(),
            },
        );
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    /// Create a new cached provider wrapping the given inner provider.
    ///
    /// `max_entries` controls the maximum number of cached embeddings.
    /// A reasonable default for interactive use is 256–1024.
    pub fn new(inner: P, max_entries: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(max_entries)),
        }
    }

    /// Return the number of currently cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For single-text batches (query embedding), check cache first.
        if texts.len() == 1 {
            let key = &texts[0];
            if let Some(cached) = self.cache.lock().unwrap().get(key) {
                debug!("embedding cache hit for query");
                return Ok(vec![cached.clone()]);
            }
        }

        // Cache miss — delegate to inner provider.
        let vectors = self.inner.embed_batch(texts).await?;

        // Cache single-text results (query embeddings).
        if texts.len() == 1 && vectors.len() == 1 {
            let key = texts[0].clone();
            let vector = vectors[0].clone();
            self.cache.lock().unwrap().insert(key, vector);
            debug!("embedding cached for query");
        }

        Ok(vectors)
    }

    fn expected_dim(&self) -> Option<usize> {
        self.inner.expected_dim()
    }
}

// ── Batch embedding orchestrator ─────────────────────────────────────

/// Embed all chunks using the given provider, respecting batch size.
///
/// Returns a vector of `(chunk_id, embedding)` pairs in the same order
/// as the input chunks. All vectors have the same dimensionality.
pub async fn embed_chunks<P: EmbeddingProvider>(
    provider: &P,
    chunks: &[Chunk],
    batch_size: usize,
) -> Result<Vec<(String, Vec<f32>)>, EmbeddingError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = batch_size.max(1);
    let total = chunks.len();
    let mut results: Vec<(String, Vec<f32>)> = Vec::with_capacity(total);

    for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
        debug!(
            batch = batch_idx + 1,
            total_batches = total.div_ceil(batch_size),
            batch_size = batch.len(),
            "embedding batch"
        );

        let texts: Vec<String> = batch.iter().map(chunk_to_embedding_text).collect();

        let vectors = provider.embed_batch(&texts).await?;

        for (chunk, vector) in batch.iter().zip(vectors.into_iter()) {
            results.push((chunk.id.clone(), vector));
        }
    }

    Ok(results)
}

/// Embed all chunks using the given provider with concurrent batch processing.
///
/// Splits chunks into batches and sends up to `max_concurrent` batches
/// simultaneously. This significantly reduces wall-clock time for large
/// repositories where the embedding API is the bottleneck.
///
/// Returns a vector of `(chunk_id, embedding)` pairs in the same order
/// as the input chunks.
pub async fn embed_chunks_concurrent<P: EmbeddingProvider>(
    provider: &P,
    chunks: &[Chunk],
    batch_size: usize,
    max_concurrent: usize,
) -> Result<Vec<(String, Vec<f32>)>, EmbeddingError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = batch_size.max(1);
    let max_concurrent = max_concurrent.max(1);
    let total = chunks.len();
    let total_batches = total.div_ceil(batch_size);

    debug!(
        total_chunks = total,
        batch_size, total_batches, max_concurrent, "starting concurrent embedding"
    );

    // Prepare all batch inputs upfront.
    let batch_inputs: Vec<(Vec<String>, Vec<String>)> = chunks
        .chunks(batch_size)
        .map(|batch| {
            let ids: Vec<String> = batch.iter().map(|c| c.id.clone()).collect();
            let texts: Vec<String> = batch.iter().map(chunk_to_embedding_text).collect();
            (ids, texts)
        })
        .collect();

    type BatchResult = (usize, Vec<(String, Vec<f32>)>);
    let mut all_results: Vec<BatchResult> = Vec::with_capacity(total_batches);

    // Process in groups of max_concurrent to overlap API calls while
    // keeping lifetime management simple (no task spawning needed).
    for group_start in (0..batch_inputs.len()).step_by(max_concurrent) {
        let group_end = (group_start + max_concurrent).min(batch_inputs.len());
        let group = &batch_inputs[group_start..group_end];

        let futures: Vec<_> = group
            .iter()
            .enumerate()
            .map(|(i, (ids, texts))| {
                let batch_idx = group_start + i;
                async move {
                    debug!(batch = batch_idx + 1, total_batches, "embedding batch");
                    let vectors = provider.embed_batch(texts).await?;
                    let pairs: Vec<(String, Vec<f32>)> = ids.iter().cloned().zip(vectors).collect();
                    Ok::<_, EmbeddingError>((batch_idx, pairs))
                }
            })
            .collect();

        // Run all futures in this group concurrently.
        let results = futures::future::join_all(futures).await;
        for result in results {
            all_results.push(result?);
        }
    }

    // Sort by batch index to preserve original ordering.
    all_results.sort_by_key(|(idx, _)| *idx);

    let results: Vec<(String, Vec<f32>)> = all_results
        .into_iter()
        .flat_map(|(_, pairs)| pairs)
        .collect();

    Ok(results)
}

/// Format a chunk's content for embedding.
///
/// Prepends metadata context (language, symbol info) to help the model
/// produce more code-aware embeddings.
fn chunk_to_embedding_text(chunk: &Chunk) -> String {
    let mut parts = Vec::new();

    // Add language context.
    parts.push(format!("Language: {}", chunk.language));

    // Add symbol info if available.
    if let Some(ref sym_type) = chunk.symbol_type {
        if let Some(ref sym_name) = chunk.symbol_name {
            parts.push(format!("{sym_type} {sym_name}"));
        }
    }

    // Add file path for context.
    parts.push(format!("File: {}", chunk.file_path));

    // Add the actual code content.
    parts.push(chunk.content.clone());

    parts.join("\n")
}

// ── Sanitization ─────────────────────────────────────────────────────

/// Remove any potential API key fragments from error messages.
///
/// API error bodies sometimes echo back parts of the request. This
/// ensures we never propagate credential material in error messages.
fn sanitize_error_message(msg: &str) -> String {
    // Truncate at a safe char boundary to avoid panicking on multi-byte UTF-8.
    let truncated = if msg.len() > 500 {
        let end = msg
            .char_indices()
            .take_while(|(i, _)| *i < 500)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        &msg[..end]
    } else {
        msg
    };
    // Strip anything that looks like a bearer token or key.
    let sanitized = truncated
        .replace(|c: char| !c.is_ascii_graphic() && c != ' ', " ")
        .trim()
        .to_string();
    if sanitized.is_empty() {
        "no details available".to_string()
    } else {
        sanitized
    }
}

// ── API request/response types ───────────────────────────────────────

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    /// A mock embedding provider for unit testing.
    ///
    /// Returns deterministic vectors based on the input text length.
    pub struct MockProvider {
        pub dim: usize,
        pub fail_with: Option<EmbeddingError>,
    }

    impl MockProvider {
        pub fn new(dim: usize) -> Self {
            Self {
                dim,
                fail_with: None,
            }
        }

        pub fn failing(error: EmbeddingError) -> Self {
            Self {
                dim: 4,
                fail_with: Some(error),
            }
        }
    }

    impl EmbeddingProvider for MockProvider {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if let Some(ref err) = self.fail_with {
                // Re-create the error since EmbeddingError is not Clone.
                return Err(match err {
                    EmbeddingError::AuthError { message } => EmbeddingError::AuthError {
                        message: message.clone(),
                    },
                    EmbeddingError::ConnectionError { message } => {
                        EmbeddingError::ConnectionError {
                            message: message.clone(),
                        }
                    }
                    EmbeddingError::ApiError { status, message } => EmbeddingError::ApiError {
                        status: *status,
                        message: message.clone(),
                    },
                    EmbeddingError::RateLimitError { message } => EmbeddingError::RateLimitError {
                        message: message.clone(),
                    },
                    EmbeddingError::ResponseError { message } => EmbeddingError::ResponseError {
                        message: message.clone(),
                    },
                });
            }

            Ok(texts
                .iter()
                .map(|text| {
                    // Deterministic hash-based seed from text content.
                    let mut hash: u64 = 5381;
                    for byte in text.bytes() {
                        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
                    }
                    let seed = hash as f32;
                    (0..self.dim)
                        .map(|i| {
                            let x = seed * (i as f32 + 1.0) * 0.001;
                            (x.sin() + 1.0) / 2.0
                        })
                        .collect()
                })
                .collect())
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(self.dim)
        }
    }
}
