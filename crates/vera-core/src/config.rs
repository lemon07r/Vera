//! Configuration types and defaults for Vera's pipeline.

use serde::{Deserialize, Serialize};

/// Top-level configuration for Vera.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VeraConfig {
    /// Indexing configuration.
    pub indexing: IndexingConfig,
    /// Retrieval configuration.
    pub retrieval: RetrievalConfig,
    /// Embedding configuration.
    pub embedding: EmbeddingConfig,
}

/// Configuration for the indexing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Maximum lines for a single chunk before splitting.
    pub max_chunk_lines: u32,
    /// Default path exclusion patterns (in addition to .gitignore).
    pub default_excludes: Vec<String>,
    /// Maximum file size in bytes to index (skip larger files).
    pub max_file_size_bytes: u64,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            max_chunk_lines: 200,
            default_excludes: vec![
                ".git".to_string(),
                ".vera".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "build".to_string(),
                "dist".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
            ],
            max_file_size_bytes: 1_000_000, // 1MB
        }
    }
}

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return by default.
    pub default_limit: usize,
    /// RRF fusion constant (k in 1/(k + rank)).
    pub rrf_k: f64,
    /// Number of candidates to pass to the reranker.
    pub rerank_candidates: usize,
    /// Whether to enable reranking (requires API credentials).
    pub reranking_enabled: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            rrf_k: 60.0,
            rerank_candidates: 50,
            reranking_enabled: true,
        }
    }
}

/// Configuration for the embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Batch size for embedding API calls.
    pub batch_size: usize,
    /// Maximum number of concurrent embedding API requests.
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
    /// Maximum stored vector dimensionality.
    ///
    /// If the embedding model produces vectors larger than this, they
    /// are truncated to this dimensionality before storage. Qwen3 models
    /// support Matryoshka-style truncation, so lower dimensions still
    /// yield good retrieval quality while dramatically reducing index size.
    /// Set to 0 to store full-dimensionality vectors.
    pub max_stored_dim: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let is_local = is_local_mode();
        Self {
            batch_size: if is_local { 4 } else { 128 },
            max_concurrent_requests: if is_local { 1 } else { 8 },
            timeout_secs: 60,
            max_retries: 3,
            max_stored_dim: 1024,
        }
    }
}

/// Check if the local inference mode is active.
pub fn is_local_mode() -> bool {
    std::env::var("VERA_LOCAL")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = VeraConfig::default();
        assert!(config.indexing.max_chunk_lines > 0);
        assert!(config.retrieval.default_limit > 0);
        assert!(config.retrieval.rrf_k > 0.0);
        assert!(config.embedding.batch_size > 0);
    }

    #[test]
    fn config_serialization_round_trip() {
        let config = VeraConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VeraConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.indexing.max_chunk_lines,
            config.indexing.max_chunk_lines
        );
        assert_eq!(
            deserialized.retrieval.default_limit,
            config.retrieval.default_limit
        );
    }

    #[test]
    fn default_excludes_contains_common_dirs() {
        let config = IndexingConfig::default();
        assert!(config.default_excludes.contains(&".git".to_string()));
        assert!(
            config
                .default_excludes
                .contains(&"node_modules".to_string())
        );
        assert!(config.default_excludes.contains(&"target".to_string()));
    }
}
