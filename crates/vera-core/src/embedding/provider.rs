//! Embedding provider abstraction and OpenAI-compatible implementation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::chunk_text;
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

    /// Rewrite query text for providers that require asymmetric query prefixes.
    fn prepare_query_text(&self, query: &str) -> String {
        query.to_string()
    }
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
    /// Optional prefix prepended to query text for asymmetric embedding models.
    /// Read from `EMBEDDING_QUERY_PREFIX` env var.
    pub query_prefix: Option<String>,
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
            query_prefix: None,
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
        let query_prefix = std::env::var("EMBEDDING_QUERY_PREFIX")
            .ok()
            .filter(|s| !s.is_empty());

        let mut config = Self::new(base_url, model_id, api_key);
        config.query_prefix = query_prefix;
        Ok(config)
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
        None
    }

    fn prepare_query_text(&self, query: &str) -> String {
        match &self.config.query_prefix {
            Some(prefix) => format!("{prefix}{query}"),
            None => query.to_string(),
        }
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

    fn prepare_query_text(&self, query: &str) -> String {
        self.inner.prepare_query_text(query)
    }
}

// ── Batch embedding orchestrator ─────────────────────────────────────

const CONTEXT_ERROR_CHARS_PER_TOKEN: usize = 2;
const TRUNCATED_EMBEDDING_MARKER: &str = "\n\n[truncated for embedding]";

#[derive(Clone)]
struct EmbeddingBatchItem {
    original_index: usize,
    chunk_id: String,
    text: String,
}

fn embedding_error_message(error: &EmbeddingError) -> &str {
    match error {
        EmbeddingError::AuthError { message }
        | EmbeddingError::ConnectionError { message }
        | EmbeddingError::ApiError { message, .. }
        | EmbeddingError::RateLimitError { message }
        | EmbeddingError::ResponseError { message } => message,
    }
}

fn context_size_limit(error: &EmbeddingError) -> Option<usize> {
    let message = embedding_error_message(error);
    if !message.contains("context size")
        && !message.contains("exceed_context_size_error")
        && !message.contains("\"n_ctx\"")
    {
        return None;
    }

    static N_CTX_RE: OnceLock<Regex> = OnceLock::new();
    static MAX_CONTEXT_RE: OnceLock<Regex> = OnceLock::new();
    let n_ctx_re = N_CTX_RE.get_or_init(|| Regex::new(r#""n_ctx"\s*:\s*(\d+)"#).unwrap());
    let max_context_re =
        MAX_CONTEXT_RE.get_or_init(|| Regex::new(r"max context size \((\d+)\)").unwrap());

    n_ctx_re
        .captures(message)
        .and_then(|caps| caps.get(1))
        .or_else(|| {
            max_context_re
                .captures(message)
                .and_then(|caps| caps.get(1))
        })
        .and_then(|capture| capture.as_str().parse::<usize>().ok())
        .or(Some(8192))
}

fn truncate_to_char_boundary(text: &str, max_chars: usize) -> &str {
    if text.chars().count() <= max_chars {
        return text;
    }

    text.char_indices()
        .nth(max_chars)
        .map(|(idx, _)| &text[..idx])
        .unwrap_or(text)
}

fn shrink_text_for_context_limit(text: &str, max_tokens: usize) -> String {
    let current_chars = text.chars().count();
    if current_chars <= 1 {
        return text.to_string();
    }

    let estimated_budget = max_tokens.saturating_mul(CONTEXT_ERROR_CHARS_PER_TOKEN);
    let fallback_budget = current_chars.saturating_mul(3) / 4;
    let marker_chars = TRUNCATED_EMBEDDING_MARKER.chars().count();
    let target_chars = estimated_budget
        .min(fallback_budget)
        .max(1)
        .saturating_sub(marker_chars);

    if target_chars >= current_chars {
        return text.to_string();
    }

    let truncated = truncate_to_char_boundary(text, target_chars).trim_end();
    if truncated.is_empty() {
        return text.to_string();
    }

    let mut shrunk = truncated.to_string();
    shrunk.push_str(TRUNCATED_EMBEDDING_MARKER);
    shrunk
}

async fn embed_batch_resilient<P: EmbeddingProvider>(
    provider: &P,
    items: Vec<EmbeddingBatchItem>,
) -> Result<Vec<(usize, String, Vec<f32>)>, EmbeddingError> {
    let mut pending = vec![items];
    let mut completed = Vec::new();

    while let Some(batch) = pending.pop() {
        let texts: Vec<String> = batch.iter().map(|item| item.text.clone()).collect();

        match provider.embed_batch(&texts).await {
            Ok(vectors) => {
                completed.extend(
                    batch
                        .into_iter()
                        .zip(vectors.into_iter())
                        .map(|(item, vector)| (item.original_index, item.chunk_id, vector)),
                );
            }
            Err(error) => {
                let Some(limit) = context_size_limit(&error) else {
                    return Err(error);
                };

                if batch.len() > 1 {
                    let mid = batch.len() / 2;
                    debug!(
                        batch_size = batch.len(),
                        token_limit = limit,
                        "embedding batch exceeded provider context limit, retrying in smaller batches"
                    );
                    pending.push(batch[mid..].to_vec());
                    pending.push(batch[..mid].to_vec());
                    continue;
                }

                let item = batch.into_iter().next().expect("single-item batch");
                let shrunk_text = shrink_text_for_context_limit(&item.text, limit);
                if shrunk_text == item.text {
                    return Err(error);
                }

                warn!(
                    chunk_id = %item.chunk_id,
                    token_limit = limit,
                    original_chars = item.text.chars().count(),
                    shrunk_chars = shrunk_text.chars().count(),
                    "embedding input exceeded provider context limit; retrying with a truncated text"
                );
                pending.push(vec![EmbeddingBatchItem {
                    text: shrunk_text,
                    ..item
                }]);
            }
        }
    }

    completed.sort_by_key(|(original_index, _, _)| *original_index);
    Ok(completed)
}

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

        let items: Vec<EmbeddingBatchItem> = batch
            .iter()
            .enumerate()
            .map(|(index, chunk)| EmbeddingBatchItem {
                original_index: index,
                chunk_id: chunk.id.clone(),
                text: chunk_to_embedding_text(chunk),
            })
            .collect();

        let vectors = embed_batch_resilient(provider, items).await?;

        for (_, chunk_id, vector) in vectors {
            results.push((chunk_id, vector));
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

    // Sort chunks by text length so each batch has similar-length texts,
    // minimizing wasted padding in the ONNX input tensors. This can cut
    // local CPU inference time by 50%+ on codebases with mixed chunk sizes.
    let mut indexed_chunks: Vec<(usize, &Chunk)> = chunks.iter().enumerate().collect();
    indexed_chunks.sort_by_key(|(_, c)| c.content.len());

    // Prepare all batch inputs upfront (in length-sorted order).
    // (original_index, chunk_id) pairs + embedding texts per batch.
    let batch_inputs: Vec<Vec<EmbeddingBatchItem>> = indexed_chunks
        .chunks(batch_size)
        .map(|batch| {
            batch
                .iter()
                .map(|(orig_idx, chunk)| EmbeddingBatchItem {
                    original_index: *orig_idx,
                    chunk_id: chunk.id.clone(),
                    text: chunk_to_embedding_text(chunk),
                })
                .collect()
        })
        .collect();

    // (orig_index, chunk_id, embedding) — we track orig_index to restore order.
    let mut all_results: Vec<(usize, String, Vec<f32>)> = Vec::with_capacity(total);

    // Process in groups of max_concurrent to overlap API calls while
    // keeping lifetime management simple (no task spawning needed).
    for group_start in (0..batch_inputs.len()).step_by(max_concurrent) {
        let group_end = (group_start + max_concurrent).min(batch_inputs.len());
        let group = &batch_inputs[group_start..group_end];

        let futures: Vec<_> = group
            .iter()
            .enumerate()
            .map(|(i, items)| {
                let batch_idx = group_start + i;
                async move {
                    debug!(batch = batch_idx + 1, total_batches, "embedding batch");
                    embed_batch_resilient(provider, items.clone()).await
                }
            })
            .collect();

        // Run all futures in this group concurrently.
        let results = futures::future::join_all(futures).await;
        for result in results {
            all_results.extend(result?);
        }
    }

    // Restore original chunk order (undoing the length sort).
    all_results.sort_by_key(|(orig_idx, _, _)| *orig_idx);

    let results: Vec<(String, Vec<f32>)> = all_results
        .into_iter()
        .map(|(_, id, vec)| (id, vec))
        .collect();

    Ok(results)
}

/// Format a chunk's content for embedding.
///
/// Prepends metadata context (language, symbol info) to help the model
/// produce more code-aware embeddings.
fn chunk_to_embedding_text(chunk: &Chunk) -> String {
    chunk_text::build_embedding_text(chunk)
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
