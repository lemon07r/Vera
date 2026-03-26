//! Configuration types and defaults for Vera's pipeline.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

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

/// ONNX execution provider for local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    Rocm,
    DirectMl,
}

impl fmt::Display for OnnxExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Rocm => write!(f, "rocm"),
            Self::DirectMl => write!(f, "directml"),
        }
    }
}

/// Inference backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBackend {
    /// Use external OpenAI-compatible API for embeddings/reranking.
    Api,
    /// Use local Jina ONNX models with the specified execution provider.
    OnnxJina(OnnxExecutionProvider),
}

impl InferenceBackend {
    /// True if this backend uses local ONNX inference.
    pub fn is_local(self) -> bool {
        matches!(self, Self::OnnxJina(_))
    }

    /// Get the execution provider (only for local backends).
    pub fn execution_provider(self) -> Option<OnnxExecutionProvider> {
        match self {
            Self::OnnxJina(ep) => Some(ep),
            Self::Api => None,
        }
    }
}

impl fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api => write!(f, "api"),
            Self::OnnxJina(ep) => write!(f, "onnx-jina-{ep}"),
        }
    }
}

impl FromStr for InferenceBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "api" => Ok(Self::Api),
            "onnx-jina-cpu" => Ok(Self::OnnxJina(OnnxExecutionProvider::Cpu)),
            "onnx-jina-cuda" => Ok(Self::OnnxJina(OnnxExecutionProvider::Cuda)),
            "onnx-jina-rocm" => Ok(Self::OnnxJina(OnnxExecutionProvider::Rocm)),
            "onnx-jina-directml" => Ok(Self::OnnxJina(OnnxExecutionProvider::DirectMl)),
            other => Err(format!("unknown backend: {other}")),
        }
    }
}

/// Check if the local inference mode is active (legacy env var support).
pub fn is_local_mode() -> bool {
    std::env::var("VERA_LOCAL")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

impl VeraConfig {
    /// Adjust embedding parameters to match the actual backend.
    ///
    /// Saved configs may have API-mode defaults (batch 128, concurrency 8)
    /// even when the user switches to local mode. CPU inference needs small
    /// batches; GPU can handle larger ones.
    pub fn adjust_for_backend(&mut self, backend: InferenceBackend) {
        match backend {
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => {
                self.embedding.batch_size = self.embedding.batch_size.min(4);
                self.embedding.max_concurrent_requests = self.embedding.max_concurrent_requests.min(1);
            }
            InferenceBackend::OnnxJina(_) => {
                // GPU EPs benefit from larger batches but shouldn't exceed 32
                // (VRAM-limited for the nano model). Single concurrent request
                // since the GPU is already saturated by one batch.
                self.embedding.batch_size = self.embedding.batch_size.min(32);
                self.embedding.max_concurrent_requests = self.embedding.max_concurrent_requests.min(1);
            }
            InferenceBackend::Api => {}
        }
    }
}

/// Resolve the effective inference backend from a CLI flag or environment.
pub fn resolve_backend(backend: Option<InferenceBackend>) -> InferenceBackend {
    if let Some(b) = backend {
        return b;
    }
    // Legacy: VERA_LOCAL=1 maps to onnx-jina-cpu
    if is_local_mode() {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu);
    }
    InferenceBackend::Api
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
