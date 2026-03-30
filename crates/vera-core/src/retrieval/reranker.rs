//! Cross-encoder reranker for refining retrieval results.
//!
//! After hybrid retrieval produces candidates via RRF fusion, the reranker
//! sends the top-N candidates along with the query to a cross-encoder API
//! (e.g. Qwen3-Reranker via SiliconFlow). The API scores each query-document
//! pair independently, producing more accurate relevance scores than the
//! fast-but-approximate RRF fusion.
//!
//! Graceful degradation: if the reranker API is unavailable (timeout, 5xx,
//! connection error), the pipeline returns unreranked results with a warning.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::chunk_text::{file_name, normalize_path_tokens};
use crate::retrieval::ranking::file_role_label;
use crate::types::SearchResult;

// ── Error types ──────────────────────────────────────────────────────

/// Errors specific to the reranking pipeline.
#[derive(Debug, thiserror::Error)]
pub enum RerankerError {
    /// Authentication failure (invalid or missing API key).
    #[error("reranker API authentication failed: {message}")]
    AuthError { message: String },

    /// Cannot reach the reranker endpoint.
    #[error("reranker API connection failed: {message}")]
    ConnectionError { message: String },

    /// The API returned a non-auth, non-connection error.
    #[error("reranker API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    /// Rate limit exceeded.
    #[error("reranker API rate limit exceeded: {message}")]
    RateLimitError { message: String },

    /// Unexpected response format.
    #[error("unexpected reranker API response: {message}")]
    ResponseError { message: String },
}

// ── Reranker trait ───────────────────────────────────────────────────

/// Trait abstracting a reranker provider.
///
/// Implementations take a query and a set of document texts, returning
/// relevance scores for each document. The scores are used to reorder
/// search results after initial retrieval.
#[allow(async_fn_in_trait)]
pub trait Reranker: Send + Sync {
    /// Score each document against the query.
    ///
    /// Returns a vector of `(original_index, relevance_score)` pairs,
    /// sorted by relevance_score descending.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError>;
}

/// A single reranking score for a document.
#[derive(Debug, Clone)]
pub struct RerankScore {
    /// Original index in the input documents array.
    pub index: usize,
    /// Relevance score from the reranker (higher is better).
    pub relevance_score: f64,
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for an API-based reranker.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Base URL for the API (e.g. "https://api.siliconflow.com/v1").
    pub base_url: String,
    /// Model identifier (e.g. "Qwen/Qwen3-Reranker-8B").
    pub model_id: String,
    /// API key (never logged or exposed).
    api_key: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
    /// Maximum number of documents to send per `/rerank` request.
    ///
    /// Some backends (notably llama.cpp reranking models) have a small
    /// context window and reject large document arrays in a single call.
    pub max_docs_per_request: usize,
    /// Maximum characters per formatted reranker document.
    ///
    /// Long code snippets are split into overlapping windows (and truncated
    /// as a fallback) before reranking to avoid model context overflows on
    /// small rerank models.
    pub max_document_chars: usize,
}

impl RerankerConfig {
    /// Create a new config. The API key is stored opaquely and never exposed.
    pub fn new(base_url: String, model_id: String, api_key: String) -> Self {
        Self {
            base_url,
            model_id,
            api_key,
            timeout: Duration::from_secs(30),
            max_retries: 2,
            max_docs_per_request: 8,
            max_document_chars: 1200,
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads:
    /// - `RERANKER_MODEL_BASE_URL`
    /// - `RERANKER_MODEL_ID`
    /// - `RERANKER_MODEL_API_KEY`
    /// - `RERANKER_MAX_DOCS_PER_REQUEST` (optional)
    /// - `RERANKER_MAX_DOCUMENT_CHARS` (optional)
    pub fn from_env() -> Result<Self> {
        let base_url =
            std::env::var("RERANKER_MODEL_BASE_URL").context("RERANKER_MODEL_BASE_URL not set")?;
        let model_id = std::env::var("RERANKER_MODEL_ID").context("RERANKER_MODEL_ID not set")?;
        let api_key =
            std::env::var("RERANKER_MODEL_API_KEY").context("RERANKER_MODEL_API_KEY not set")?;

        let mut cfg = Self::new(base_url, model_id, api_key);
        if let Ok(value) = std::env::var("RERANKER_MAX_DOCS_PER_REQUEST") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.max_docs_per_request = parsed.max(1);
            }
        }
        if let Ok(value) = std::env::var("RERANKER_MAX_DOCUMENT_CHARS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.max_document_chars = parsed.max(64);
            }
        }

        Ok(cfg)
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

    /// Set maximum documents per API request.
    pub fn with_max_docs_per_request(mut self, max_docs_per_request: usize) -> Self {
        self.max_docs_per_request = max_docs_per_request.max(1);
        self
    }

    /// Set maximum characters per formatted reranker document.
    pub fn with_max_document_chars(mut self, max_document_chars: usize) -> Self {
        self.max_document_chars = max_document_chars.max(64);
        self
    }
}

// ── API-based reranker ───────────────────────────────────────────────

/// Reranker that calls an external cross-encoder API (SiliconFlow-compatible).
///
/// Sends query + documents to the `/rerank` endpoint and returns scored
/// results. Compatible with SiliconFlow, Jina, Cohere, and similar APIs
/// that implement the `/v1/rerank` endpoint format.
pub struct ApiReranker {
    client: reqwest::Client,
    config: RerankerConfig,
}

impl ApiReranker {
    /// Create a new API-based reranker from configuration.
    pub fn new(config: RerankerConfig) -> Result<Self> {
        crate::init_tls();
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client for reranker")?;

        Ok(Self { client, config })
    }

    /// Build the rerank endpoint URL.
    fn endpoint_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/rerank")
    }

    /// Execute a rerank API call with retry logic.
    async fn call_api(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        let url = self.endpoint_url();
        let body = RerankRequest {
            model: &self.config.model_id,
            query,
            documents,
            top_n: Some(top_n),
            return_documents: Some(false),
        };

        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt.min(4) - 1));
                debug!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying reranker API"
                );
                tokio::time::sleep(delay).await;
            }

            match self.send_request(&url, &body).await {
                Ok(scores) => return Ok(scores),
                Err(e) => {
                    // Don't retry auth errors.
                    if matches!(e, RerankerError::AuthError { .. }) {
                        return Err(e);
                    }
                    warn!(
                        attempt = attempt + 1,
                        max = self.config.max_retries + 1,
                        error = %e,
                        "reranker API call failed"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RerankerError::ApiError {
            status: 0,
            message: "all retries exhausted".to_string(),
        }))
    }

    /// Send a single HTTP request and parse the response.
    async fn send_request(
        &self,
        url: &str,
        body: &RerankRequest<'_>,
    ) -> Result<Vec<RerankScore>, RerankerError> {
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
                    RerankerError::ConnectionError {
                        message: format!("failed to connect to reranker API: {e}"),
                    }
                } else {
                    RerankerError::ConnectionError {
                        message: format!("request failed: {e}"),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if status == 401 || status == 403 {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::AuthError {
                message: sanitize_error_message(&text),
            });
        }

        if status == 429 {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::RateLimitError {
                message: sanitize_error_message(&text),
            });
        }

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::ApiError {
                status,
                message: sanitize_error_message(&text),
            });
        }

        let resp: RerankResponse =
            response
                .json()
                .await
                .map_err(|e| RerankerError::ResponseError {
                    message: format!("failed to parse reranker response: {e}"),
                })?;

        // Convert to RerankScore, sorted by relevance_score descending.
        let mut scores: Vec<RerankScore> = resp
            .results
            .into_iter()
            .map(|r| RerankScore {
                index: r.index,
                relevance_score: r.relevance_score,
            })
            .collect();

        scores.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scores)
    }

    /// Rerank documents in adaptive batches, handling context limits.
    ///
    /// Splits documents into batches of max_docs_per_request. If a batch fails
    /// with a context limit error, recursively splits it in half and retries.
    /// For single-document failures, progressively shrinks the document until
    /// it fits or gives up.
    async fn rerank_adaptive_batches(
        &self,
        query: &str,
        documents: &[String],
        base_index: usize,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let max_docs = self.config.max_docs_per_request.max(1);
        let mut pending: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < documents.len() {
            let end = (start + max_docs).min(documents.len());
            pending.push((start, end));
            start = end;
        }

        let mut out = Vec::with_capacity(documents.len());
        while let Some((start, end)) = pending.pop() {
            let slice = &documents[start..end];
            match self.call_api(query, slice, slice.len()).await {
                Ok(mut scores) => {
                    for score in &mut scores {
                        score.index += base_index + start;
                    }
                    out.extend(scores);
                }
                Err(err) if is_context_limit_error(&err) && slice.len() > 1 => {
                    // Split this failing batch and retry each half.
                    let mid = start + slice.len() / 2;
                    pending.push((mid, end));
                    pending.push((start, mid));
                }
                Err(err) if is_context_limit_error(&err) && slice.len() == 1 => {
                    // Even one document overflowed. Retry with progressively smaller text.
                    let original = &documents[start];
                    let mut budget = (self.config.max_document_chars / 2).max(64);
                    let mut recovered = None;

                    while budget >= 64 {
                        let shrunk = truncate_reranker_document(original, budget);
                        let singleton = vec![shrunk];
                        match self.call_api(query, &singleton, 1).await {
                            Ok(mut scores) => {
                                for score in &mut scores {
                                    score.index += base_index + start;
                                }
                                recovered = Some(scores);
                                warn!(
                                    budget,
                                    document_index = base_index + start,
                                    "reranker context overflow recovered by shrinking single document"
                                );
                                break;
                            }
                            Err(e) if is_context_limit_error(&e) => {
                                budget /= 2;
                            }
                            Err(e) => return Err(e),
                        }
                    }

                    if let Some(scores) = recovered {
                        out.extend(scores);
                    } else {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }
        }

        Ok(out)
    }
}

impl Reranker for ApiReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let overlap_chars = (self.config.max_document_chars / 6).clamp(32, 256);
        let mut expanded_documents = Vec::new();
        let mut expanded_to_original = Vec::new();

        for (doc_index, doc) in documents.iter().enumerate() {
            let windows = split_reranker_document(doc, self.config.max_document_chars, overlap_chars);
            for window in windows {
                expanded_to_original.push(doc_index);
                expanded_documents.push(window);
            }
        }

        debug!(
            original_docs = documents.len(),
            expanded_docs = expanded_documents.len(),
            max_docs_per_request = self.config.max_docs_per_request,
            max_document_chars = self.config.max_document_chars,
            "prepared reranker windows"
        );

        let expanded_scores = self
            .rerank_adaptive_batches(query, &expanded_documents, 0)
            .await?;
        Ok(aggregate_split_scores(
            expanded_scores,
            &expanded_to_original,
            documents.len(),
        ))
    }
}

// ── Rerank search results ────────────────────────────────────────────

/// Rerank a set of search results using a cross-encoder reranker.
///
/// Sends the top-N candidates to the reranker API, reorders them by
/// reranker scores, and returns the reordered results. The reranker
/// score replaces the original score in each result.
///
/// If the reranker fails, returns `Err` — the caller decides whether
/// to fall back to unreranked results.
pub async fn rerank_results(
    reranker: &impl Reranker,
    query: &str,
    results: &[SearchResult],
    top_n: usize,
) -> Result<Vec<SearchResult>, RerankerError> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    // Take only top_n candidates for reranking.
    let candidates: Vec<&SearchResult> = results.iter().take(top_n).collect();

    // Extract document texts for the reranker.
    let documents: Vec<String> = candidates.iter().map(|r| format_for_reranker(r)).collect();

    debug!(
        query = query,
        candidates = candidates.len(),
        "sending candidates to reranker"
    );

    // Call the reranker.
    let scores = reranker.rerank(query, &documents).await?;

    debug!(
        query = query,
        scored = scores.len(),
        "received reranker scores"
    );

    // Reorder results by reranker scores.
    let mut reranked: Vec<SearchResult> = scores
        .iter()
        .filter_map(|score| {
            if score.index < candidates.len() {
                let mut result = candidates[score.index].clone();
                result.score = score.relevance_score;
                Some(result)
            } else {
                warn!(
                    index = score.index,
                    candidates = candidates.len(),
                    "reranker returned out-of-bounds index, skipping"
                );
                None
            }
        })
        .collect();

    // Ensure results are sorted by score descending (should already be from the API).
    reranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(reranked)
}

/// Format a search result for the reranker API.
///
/// Includes metadata context to help the cross-encoder understand the code.
fn format_for_reranker(result: &SearchResult) -> String {
    let mut parts = Vec::new();
    let filename = file_name(&result.file_path);

    if let Some(ref sym_name) = result.symbol_name {
        parts.push(format!("Symbol: {sym_name}"));
    }
    if let Some(ref sym_type) = result.symbol_type {
        parts.push(format!("Symbol type: {sym_type}"));
    }

    parts.push(format!("Filename: {filename}"));
    parts.push(format!("File: {} ({})", result.file_path, result.language));
    parts.push(format!(
        "Path tokens: {}",
        normalize_path_tokens(&result.file_path)
    ));
    parts.push(format!(
        "Role: {}",
        file_role_label(&result.file_path, result.language)
    ));
    parts.push(format!("Lines: {}-{}", result.line_start, result.line_end));

    parts.push(format!("Code:\n{}", result.content));

    parts.join("\n")
}

// ── Sanitization ─────────────────────────────────────────────────────

/// Remove any potential API key fragments from error messages.
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

// ── Document splitting helpers ───────────────────────────────────────

/// Check if an error indicates a context limit overflow.
fn is_context_limit_error(error: &RerankerError) -> bool {
    let msg = match error {
        RerankerError::AuthError { message }
        | RerankerError::ConnectionError { message }
        | RerankerError::ApiError { message, .. }
        | RerankerError::RateLimitError { message }
        | RerankerError::ResponseError { message } => message,
    };
    msg.contains("context") || msg.contains("n_ctx") || msg.contains("too long")
}

/// Split a document into overlapping character windows.
///
/// Returns at least one window (the original document, possibly truncated).
fn split_reranker_document(doc: &str, max_chars: usize, overlap_chars: usize) -> Vec<String> {
    let char_count = doc.chars().count();
    if char_count <= max_chars {
        return vec![doc.to_string()];
    }

    let mut windows = Vec::new();
    let step = max_chars.saturating_sub(overlap_chars).max(1);
    let mut start = 0;

    while start < char_count {
        let end = (start + max_chars).min(char_count);
        let window: String = doc.chars().skip(start).take(end - start).collect();
        windows.push(window);

        if end >= char_count {
            break;
        }
        start += step;
    }

    if windows.is_empty() {
        vec![truncate_reranker_document(doc, max_chars)]
    } else {
        windows
    }
}

/// Truncate a document to a maximum character count.
fn truncate_reranker_document(doc: &str, max_chars: usize) -> String {
    let char_count = doc.chars().count();
    if char_count <= max_chars {
        return doc.to_string();
    }
    doc.chars().take(max_chars).collect()
}

/// Aggregate scores from split document windows back to original documents.
///
/// For each original document, takes the maximum score from all its windows.
fn aggregate_split_scores(
    expanded_scores: Vec<RerankScore>,
    expanded_to_original: &[usize],
    original_count: usize,
) -> Vec<RerankScore> {
    let mut best_scores: Vec<Option<f64>> = vec![None; original_count];

    for score in expanded_scores {
        if score.index < expanded_to_original.len() {
            let original_index = expanded_to_original[score.index];
            if original_index < original_count {
                let current = best_scores[original_index];
                best_scores[original_index] = Some(
                    current
                        .map(|c| c.max(score.relevance_score))
                        .unwrap_or(score.relevance_score),
                );
            }
        }
    }

    let mut out: Vec<RerankScore> = best_scores
        .into_iter()
        .enumerate()
        .filter_map(|(idx, score)| {
            score.map(|s| RerankScore {
                index: idx,
                relevance_score: s,
            })
        })
        .collect();

    out.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    out
}

// ── API request/response types ───────────────────────────────────────

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    #[serde(alias = "score")]
    relevance_score: f64,
}

// ── Test helpers ─────────────────────────────────────────────────────

/// Mock reranker for unit testing.
///
/// Returns scores based on document length or simulates errors.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    /// A mock reranker that returns deterministic scores.
    ///
    /// Scores each document based on a simple heuristic (shorter documents
    /// score higher, simulating "precision" preference). Can also be
    /// configured to fail with a specific error.
    pub struct MockReranker {
        pub fail_with: Option<RerankerError>,
    }

    impl MockReranker {
        pub fn new() -> Self {
            Self { fail_with: None }
        }

        pub fn failing(error: RerankerError) -> Self {
            Self {
                fail_with: Some(error),
            }
        }
    }

    impl Reranker for MockReranker {
        async fn rerank(
            &self,
            _query: &str,
            documents: &[String],
        ) -> Result<Vec<RerankScore>, RerankerError> {
            if let Some(ref err) = self.fail_with {
                return Err(match err {
                    RerankerError::AuthError { message } => RerankerError::AuthError {
                        message: message.clone(),
                    },
                    RerankerError::ConnectionError { message } => RerankerError::ConnectionError {
                        message: message.clone(),
                    },
                    RerankerError::ApiError { status, message } => RerankerError::ApiError {
                        status: *status,
                        message: message.clone(),
                    },
                    RerankerError::RateLimitError { message } => RerankerError::RateLimitError {
                        message: message.clone(),
                    },
                    RerankerError::ResponseError { message } => RerankerError::ResponseError {
                        message: message.clone(),
                    },
                });
            }

            // Deterministic scoring: reverse order of input (last doc scores highest).
            let total = documents.len();
            let mut scores: Vec<RerankScore> = documents
                .iter()
                .enumerate()
                .map(|(i, _)| RerankScore {
                    index: i,
                    relevance_score: (total - i) as f64 / total as f64,
                })
                .collect();

            scores.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            Ok(scores)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, SymbolType};

    /// Helper to create a SearchResult with given parameters.
    fn make_result(
        file: &str,
        line_start: u32,
        line_end: u32,
        score: f64,
        symbol_name: Option<&str>,
        content: &str,
    ) -> SearchResult {
        SearchResult {
            file_path: file.to_string(),
            line_start,
            line_end,
            content: content.to_string(),
            language: Language::Rust,
            score,
            symbol_name: symbol_name.map(|s| s.to_string()),
            symbol_type: Some(SymbolType::Function),
        }
    }

    // ── RerankerConfig tests ─────────────────────────────────────────

    #[test]
    fn config_from_values() {
        let config = RerankerConfig::new(
            "https://api.example.com/v1".to_string(),
            "model-1".to_string(),
            "key-123".to_string(),
        );
        assert_eq!(config.base_url, "https://api.example.com/v1");
        assert_eq!(config.model_id, "model-1");
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 2);
    }

    #[test]
    fn config_with_timeout() {
        let config = RerankerConfig::new(
            "https://api.example.com/v1".to_string(),
            "model-1".to_string(),
            "key-123".to_string(),
        )
        .with_timeout(Duration::from_secs(10));
        assert_eq!(config.timeout, Duration::from_secs(10));
    }

    #[test]
    fn config_with_max_retries() {
        let config = RerankerConfig::new(
            "https://api.example.com/v1".to_string(),
            "model-1".to_string(),
            "key-123".to_string(),
        )
        .with_max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    // ── MockReranker tests ───────────────────────────────────────────

    #[tokio::test]
    async fn mock_reranker_returns_scores_for_all_documents() {
        let reranker = test_helpers::MockReranker::new();
        let docs = vec![
            "doc 1".to_string(),
            "doc 2".to_string(),
            "doc 3".to_string(),
        ];

        let scores = reranker.rerank("query", &docs).await.unwrap();

        assert_eq!(scores.len(), 3);
    }

    #[tokio::test]
    async fn mock_reranker_scores_are_descending() {
        let reranker = test_helpers::MockReranker::new();
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let scores = reranker.rerank("query", &docs).await.unwrap();

        for i in 1..scores.len() {
            assert!(
                scores[i - 1].relevance_score >= scores[i].relevance_score,
                "scores must be descending"
            );
        }
    }

    #[tokio::test]
    async fn mock_reranker_empty_documents() {
        let reranker = test_helpers::MockReranker::new();

        let scores = reranker.rerank("query", &[]).await.unwrap();

        assert!(scores.is_empty());
    }

    #[tokio::test]
    async fn mock_reranker_connection_error() {
        let reranker = test_helpers::MockReranker::failing(RerankerError::ConnectionError {
            message: "timeout".to_string(),
        });

        let result = reranker.rerank("query", &["doc".to_string()]).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RerankerError::ConnectionError { .. }
        ));
    }

    #[tokio::test]
    async fn mock_reranker_auth_error() {
        let reranker = test_helpers::MockReranker::failing(RerankerError::AuthError {
            message: "invalid key".to_string(),
        });

        let result = reranker.rerank("query", &["doc".to_string()]).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RerankerError::AuthError { .. }
        ));
    }

    // ── rerank_results tests ─────────────────────────────────────────

    #[tokio::test]
    async fn rerank_results_reorders_by_reranker_scores() {
        let reranker = test_helpers::MockReranker::new();
        let results = vec![
            make_result("a.rs", 1, 10, 0.5, Some("func_a"), "fn func_a() {}"),
            make_result("b.rs", 1, 10, 0.4, Some("func_b"), "fn func_b() {}"),
            make_result("c.rs", 1, 10, 0.3, Some("func_c"), "fn func_c() {}"),
        ];

        let reranked = rerank_results(&reranker, "test query", &results, 10)
            .await
            .unwrap();

        assert_eq!(reranked.len(), 3);
        // Scores should be descending.
        for i in 1..reranked.len() {
            assert!(
                reranked[i - 1].score >= reranked[i].score,
                "reranked scores must be descending"
            );
        }
    }

    #[tokio::test]
    async fn rerank_results_replaces_original_scores() {
        let reranker = test_helpers::MockReranker::new();
        let results = vec![
            make_result("a.rs", 1, 10, 100.0, None, "fn a() {}"),
            make_result("b.rs", 1, 10, 50.0, None, "fn b() {}"),
        ];

        let reranked = rerank_results(&reranker, "query", &results, 10)
            .await
            .unwrap();

        // Original scores (100.0, 50.0) should be replaced by reranker scores.
        for result in &reranked {
            assert!(
                result.score <= 1.0,
                "reranker scores should replace original (was {})",
                result.score
            );
        }
    }

    #[tokio::test]
    async fn rerank_results_preserves_metadata() {
        let reranker = test_helpers::MockReranker::new();
        let results = vec![make_result(
            "auth.rs",
            5,
            20,
            0.8,
            Some("authenticate"),
            "fn authenticate() {}",
        )];

        let reranked = rerank_results(&reranker, "auth", &results, 10)
            .await
            .unwrap();

        assert_eq!(reranked.len(), 1);
        let r = &reranked[0];
        assert_eq!(r.file_path, "auth.rs");
        assert_eq!(r.line_start, 5);
        assert_eq!(r.line_end, 20);
        assert_eq!(r.symbol_name.as_deref(), Some("authenticate"));
        assert_eq!(r.symbol_type, Some(SymbolType::Function));
        assert_eq!(r.language, Language::Rust);
        assert!(!r.content.is_empty());
    }

    #[tokio::test]
    async fn rerank_results_respects_top_n() {
        let reranker = test_helpers::MockReranker::new();
        let results = vec![
            make_result("a.rs", 1, 10, 0.9, None, "fn a() {}"),
            make_result("b.rs", 1, 10, 0.8, None, "fn b() {}"),
            make_result("c.rs", 1, 10, 0.7, None, "fn c() {}"),
            make_result("d.rs", 1, 10, 0.6, None, "fn d() {}"),
            make_result("e.rs", 1, 10, 0.5, None, "fn e() {}"),
        ];

        // Only rerank top 2 candidates.
        let reranked = rerank_results(&reranker, "query", &results, 2)
            .await
            .unwrap();

        assert_eq!(
            reranked.len(),
            2,
            "should only return top_n reranked results"
        );
    }

    #[tokio::test]
    async fn rerank_results_empty_input() {
        let reranker = test_helpers::MockReranker::new();

        let reranked = rerank_results(&reranker, "query", &[], 10).await.unwrap();

        assert!(reranked.is_empty());
    }

    #[tokio::test]
    async fn rerank_results_propagates_connection_error() {
        let reranker = test_helpers::MockReranker::failing(RerankerError::ConnectionError {
            message: "timeout".to_string(),
        });
        let results = vec![make_result("a.rs", 1, 10, 0.5, None, "fn a() {}")];

        let result = rerank_results(&reranker, "query", &results, 10).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RerankerError::ConnectionError { .. }
        ));
    }

    #[tokio::test]
    async fn rerank_results_propagates_api_error() {
        let reranker = test_helpers::MockReranker::failing(RerankerError::ApiError {
            status: 500,
            message: "internal error".to_string(),
        });
        let results = vec![make_result("a.rs", 1, 10, 0.5, None, "fn a() {}")];

        let result = rerank_results(&reranker, "query", &results, 10).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RerankerError::ApiError { status: 500, .. }
        ));
    }

    // ── format_for_reranker tests ────────────────────────────────────

    #[test]
    fn format_includes_symbol_info() {
        let result = make_result("lib.rs", 1, 10, 0.5, Some("my_func"), "fn my_func() {}");
        let formatted = format_for_reranker(&result);

        assert!(formatted.contains("Symbol: my_func"));
        assert!(formatted.contains("Symbol type: function"));
        assert!(formatted.contains("Filename: lib.rs"));
        assert!(formatted.contains("File: lib.rs"));
        assert!(formatted.contains("fn my_func() {}"));
    }

    #[test]
    fn format_without_symbol_info() {
        let mut result = make_result("lib.rs", 1, 10, 0.5, None, "some code");
        result.symbol_type = None;
        let formatted = format_for_reranker(&result);

        assert!(formatted.contains("Filename: lib.rs"));
        assert!(formatted.contains("File: lib.rs"));
        assert!(formatted.contains("some code"));
        assert!(!formatted.contains("Symbol type:"));
    }

    // ── sanitize_error_message tests ─────────────────────────────────

    #[test]
    fn sanitize_truncates_long_messages() {
        let long_msg = "a".repeat(1000);
        let sanitized = sanitize_error_message(&long_msg);
        assert!(sanitized.len() <= 500);
    }

    #[test]
    fn sanitize_multibyte_utf8_boundary() {
        // Create a string with multi-byte chars near the 500-byte boundary.
        // Each '🦀' is 4 bytes. 125 crabs = 500 bytes exactly, but place
        // the boundary right in the middle of a multi-byte sequence.
        let msg = "a".repeat(499) + "🦀"; // 499 + 4 = 503 bytes
        let sanitized = sanitize_error_message(&msg);
        // Should truncate before the crab emoji, not panic.
        assert!(sanitized.len() <= 500);
        assert!(sanitized.is_char_boundary(sanitized.len()));
    }

    #[test]
    fn sanitize_empty_message() {
        let sanitized = sanitize_error_message("");
        assert_eq!(sanitized, "no details available");
    }

    // ── ApiReranker endpoint URL tests ───────────────────────────────

    #[test]
    fn endpoint_url_builds_correctly() {
        let config = RerankerConfig::new(
            "https://api.siliconflow.com/v1".to_string(),
            "model".to_string(),
            "key".to_string(),
        );
        let reranker = ApiReranker::new(config).unwrap();
        assert_eq!(
            reranker.endpoint_url(),
            "https://api.siliconflow.com/v1/rerank"
        );
    }

    #[test]
    fn endpoint_url_strips_trailing_slash() {
        let config = RerankerConfig::new(
            "https://api.siliconflow.com/v1/".to_string(),
            "model".to_string(),
            "key".to_string(),
        );
        let reranker = ApiReranker::new(config).unwrap();
        assert_eq!(
            reranker.endpoint_url(),
            "https://api.siliconflow.com/v1/rerank"
        );
    }

    #[tokio::test]
    async fn api_reranker_unreachable_endpoint() {
        let config = RerankerConfig::new(
            "http://127.0.0.1:19999".to_string(),
            "model".to_string(),
            "key".to_string(),
        )
        .with_timeout(Duration::from_millis(500))
        .with_max_retries(0);

        let reranker = ApiReranker::new(config).unwrap();
        let result = reranker.rerank("test", &["document".to_string()]).await;

        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), RerankerError::ConnectionError { .. }),
            "unreachable endpoint should return connection error"
        );
    }
}
