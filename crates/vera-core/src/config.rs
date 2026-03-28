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
    /// Extra exclusion globs from CLI `--exclude` flags.
    #[serde(default)]
    pub extra_excludes: Vec<String>,
    /// Disable .gitignore and .veraignore parsing.
    #[serde(default)]
    pub no_ignore: bool,
    /// Disable smart default exclusions.
    #[serde(default)]
    pub no_default_excludes: bool,
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
            extra_excludes: Vec::new(),
            no_ignore: false,
            no_default_excludes: false,
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
    /// GPU memory limit in MB for ONNX CUDA sessions.
    /// 0 means no limit (ORT default: use all available VRAM).
    #[serde(default)]
    pub gpu_mem_limit_mb: u64,
    /// When true, forces conservative GPU settings (batch_size=1, low mem limit).
    #[serde(default)]
    pub low_vram: bool,
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
            gpu_mem_limit_mb: 0,
            low_vram: false,
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
    CoreMl,
    OpenVino,
}

impl fmt::Display for OnnxExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Rocm => write!(f, "rocm"),
            Self::DirectMl => write!(f, "directml"),
            Self::CoreMl => write!(f, "coreml"),
            Self::OpenVino => write!(f, "openvino"),
        }
    }
}

/// Inference backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBackend {
    /// Use external OpenAI-compatible API for embeddings/reranking.
    Api,
    /// Use local ONNX models with the specified execution provider.
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
            "onnx-jina-coreml" => Ok(Self::OnnxJina(OnnxExecutionProvider::CoreMl)),
            "onnx-jina-openvino" => Ok(Self::OnnxJina(OnnxExecutionProvider::OpenVino)),
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

fn backend_from_env() -> Option<InferenceBackend> {
    std::env::var("VERA_BACKEND")
        .ok()
        .and_then(|value| InferenceBackend::from_str(&value).ok())
}

impl VeraConfig {
    /// Adjust embedding parameters to match the actual backend.
    ///
    /// Saved configs may have API-mode defaults (batch 128, concurrency 8)
    /// even when the user switches to local mode. CPU inference needs small
    /// batches; GPU can handle larger ones. For GPU backends, this picks a
    /// coarse outer batch ceiling from available VRAM. The local ONNX provider
    /// still shapes the actual micro-batches from sequence length at runtime.
    pub fn adjust_for_backend(&mut self, backend: InferenceBackend) {
        match backend {
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => {
                self.embedding.batch_size = 4;
                self.embedding.max_concurrent_requests = 1;
            }
            InferenceBackend::OnnxJina(ep) => {
                self.embedding.max_concurrent_requests = 1;

                if self.embedding.low_vram {
                    self.embedding.batch_size = 1;
                    if self.embedding.gpu_mem_limit_mb == 0 {
                        self.embedding.gpu_mem_limit_mb = 1024;
                    }
                    tracing::info!(
                        "low-vram mode: batch_size=1, gpu_mem_limit={}MB",
                        self.embedding.gpu_mem_limit_mb
                    );
                    return;
                }

                let vram_mb = detect_gpu_vram_mb(ep);
                if let Some(vram) = vram_mb {
                    tracing::info!("detected GPU VRAM: {vram}MB");
                    // Auto-scale batch_size based on VRAM.
                    // Prioritize speed: use large batches when VRAM allows.
                    let auto_batch = if vram < 3072 {
                        4
                    } else if vram < 5120 {
                        16
                    } else if vram < 8192 {
                        32
                    } else if vram < 12288 {
                        64
                    } else {
                        128
                    };
                    self.embedding.batch_size = auto_batch;

                    // Set a conservative memory limit only for low-VRAM GPUs
                    // to prevent ORT from grabbing all VRAM. For >=8GB, no limit.
                    if self.embedding.gpu_mem_limit_mb == 0 && vram < 8192 {
                        // Use 80% of available VRAM.
                        self.embedding.gpu_mem_limit_mb = (vram as f64 * 0.8) as u64;
                        tracing::info!(
                            "auto-set gpu_mem_limit={}MB (80% of {vram}MB)",
                            self.embedding.gpu_mem_limit_mb
                        );
                    }
                } else {
                    // Could not detect VRAM; use safe defaults.
                    self.embedding.batch_size = 32;
                }
            }
            InferenceBackend::Api => {}
        }
    }
}

/// Detect available GPU VRAM in MB for the given execution provider.
///
/// For CUDA/ROCm, runs `nvidia-smi` or `rocm-smi`. Returns `None` if
/// detection fails (command not found, parse error, etc.).
fn detect_gpu_vram_mb(ep: OnnxExecutionProvider) -> Option<u64> {
    match ep {
        OnnxExecutionProvider::Cuda => detect_nvidia_vram_mb(),
        OnnxExecutionProvider::Rocm => detect_rocm_vram_mb(),
        // DirectML, CoreML, OpenVINO: no standard CLI detection.
        _ => None,
    }
}

/// Query free VRAM via `nvidia-smi`.
fn detect_nvidia_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Take the first GPU's free memory.
    stdout.lines().next()?.trim().parse::<u64>().ok()
}

/// Query total VRAM via `rocm-smi`.
fn detect_rocm_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // CSV output: header line then data. Look for free VRAM column.
    for line in stdout.lines().skip(1) {
        // Format varies; try to find a numeric MB value.
        if let Some(val) = line
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .next()
        {
            // rocm-smi reports bytes; convert to MB.
            return Some(val / (1024 * 1024));
        }
    }
    None
}

/// Resolve the effective inference backend from a CLI flag or environment.
pub fn resolve_backend(backend: Option<InferenceBackend>) -> InferenceBackend {
    if let Some(b) = backend {
        return b;
    }
    if let Some(b) = backend_from_env() {
        return b;
    }
    // Legacy: VERA_LOCAL=1 maps to onnx-jina-cpu
    if is_local_mode() {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu);
    }
    InferenceBackend::Api
}

/// Check whether two model names refer to the same model.
///
/// Model names may differ only by an org/repo prefix (e.g.
/// `"jinaai/jina-embeddings-v5-text-nano-retrieval"` vs
/// `"jina-embeddings-v5-text-nano-retrieval"`). This function
/// normalises both names by stripping everything up to and including
/// the last `/` and then comparing case-insensitively.
pub fn model_names_match(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.rsplit('/').next().unwrap_or(s).to_ascii_lowercase();
    norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env(key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn remove_env(key: &str) {
        unsafe {
            std::env::remove_var(key);
        }
    }

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
    fn openvino_backend_round_trip() {
        let backend = InferenceBackend::from_str("onnx-jina-openvino").unwrap();
        assert_eq!(
            backend,
            InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino)
        );
        assert_eq!(backend.to_string(), "onnx-jina-openvino");
        assert!(backend.is_local());
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

    #[test]
    fn resolve_backend_prefers_saved_backend_env() {
        let _guard = env_lock().lock().unwrap();
        set_env("VERA_BACKEND", "onnx-jina-cuda");
        set_env("VERA_LOCAL", "1");

        assert_eq!(
            resolve_backend(None),
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda)
        );

        remove_env("VERA_BACKEND");
        remove_env("VERA_LOCAL");
    }

    #[test]
    fn resolve_backend_falls_back_to_legacy_local_env() {
        let _guard = env_lock().lock().unwrap();
        remove_env("VERA_BACKEND");
        set_env("VERA_LOCAL", "1");

        assert_eq!(
            resolve_backend(None),
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu)
        );

        remove_env("VERA_LOCAL");
    }

    #[test]
    fn model_names_match_exact() {
        assert!(model_names_match(
            "jina-embeddings-v5",
            "jina-embeddings-v5"
        ));
    }

    #[test]
    fn model_names_match_with_org_prefix() {
        assert!(model_names_match(
            "jinaai/jina-embeddings-v5-text-nano-retrieval",
            "jina-embeddings-v5-text-nano-retrieval"
        ));
    }

    #[test]
    fn model_names_match_case_insensitive() {
        assert!(model_names_match(
            "Jina-Embeddings-V5",
            "jina-embeddings-v5"
        ));
    }

    #[test]
    fn model_names_match_different_models() {
        assert!(!model_names_match("jina-embeddings-v5", "jina-reranker-v2"));
    }
}
