use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;

const HUB_URL: &str = "https://huggingface.co";
const EMBEDDING_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
const EMBEDDING_ONNX_FILE: &str = "onnx/model_quantized.onnx";
const EMBEDDING_ONNX_DATA_FILE: &str = "onnx/model_quantized.onnx_data";
/// FP16 model for GPU backends (quantized INT8 ops lack CUDA kernels,
/// causing ORT to silently fall back to CPU).
const EMBEDDING_ONNX_GPU_FILE: &str = "onnx/model_fp16.onnx";
const EMBEDDING_ONNX_GPU_DATA_FILE: &str = "onnx/model_fp16.onnx_data";
const EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";
const EMBEDDING_DIM: usize = 768;
const EMBEDDING_MAX_LENGTH: usize = 512;

const CODERANK_EMBEDDING_REPO: &str = "Zenabius/CodeRankEmbed-onnx";
const CODERANK_QUERY_PREFIX: &str = "Represent this query for searching relevant code:";

pub const LOCAL_EMBEDDING_REPO_ENV: &str = "VERA_LOCAL_EMBEDDING_REPO";
pub const LOCAL_EMBEDDING_DIR_ENV: &str = "VERA_LOCAL_EMBEDDING_DIR";
pub const LOCAL_EMBEDDING_ONNX_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_ONNX_FILE";
pub const LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_ONNX_DATA_FILE";
pub const LOCAL_EMBEDDING_TOKENIZER_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_TOKENIZER_FILE";
pub const LOCAL_EMBEDDING_DIM_ENV: &str = "VERA_LOCAL_EMBEDDING_DIM";
pub const LOCAL_EMBEDDING_POOLING_ENV: &str = "VERA_LOCAL_EMBEDDING_POOLING";
pub const LOCAL_EMBEDDING_MAX_LENGTH_ENV: &str = "VERA_LOCAL_EMBEDDING_MAX_LENGTH";
pub const LOCAL_EMBEDDING_QUERY_PREFIX_ENV: &str = "VERA_LOCAL_EMBEDDING_QUERY_PREFIX";
pub const LEGACY_EMBEDDING_QUERY_PREFIX_ENV: &str = "VERA_EMBEDDING_QUERY_PREFIX";

const RERANKER_REPO: &str = "jinaai/jina-reranker-v2-base-multilingual";
const RERANKER_ONNX_FILE: &str = "onnx/model_quantized.onnx";
const RERANKER_TOKENIZER_FILE: &str = "tokenizer.json";

/// ONNX Runtime version to auto-download. Using 1.24.4 for CUDA 13 support.
/// The `ort` crate (rc.11) uses `load-dynamic` so any ABI-compatible ORT works.
const ORT_VERSION: &str = "1.24.4";
const DEFAULT_CUDA_MAJOR: u32 = 12;
const CUDA_13_ORT_MIN_MAJOR: u32 = 13;
const CUDA_RUNTIME_LIBRARY_PREFIXES: [&str; 3] =
    ["libcudart.so.", "libcublas.so.", "libcublasLt.so."];

/// ONNX Runtime 1.24.x dropped macOS x86_64 binaries. 1.23.2 is the last
/// release that ships `onnxruntime-osx-x86_64` archives.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const ORT_VERSION_MACOS_X86: &str = "1.23.2";

static ORT_INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalEmbeddingPooling {
    Mean,
    Cls,
}

impl fmt::Display for LocalEmbeddingPooling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Cls => write!(f, "cls"),
        }
    }
}

impl std::str::FromStr for LocalEmbeddingPooling {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            other => Err(format!(
                "invalid pooling mode: {other} (expected `mean` or `cls`)"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "kebab-case")]
pub enum LocalEmbeddingSource {
    HuggingFace { repo: String },
    Directory { path: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalEmbeddingModelConfig {
    pub source: LocalEmbeddingSource,
    pub onnx_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_data_file: Option<String>,
    pub tokenizer_file: String,
    pub embedding_dim: usize,
    pub pooling: LocalEmbeddingPooling,
    #[serde(default = "default_embedding_max_length")]
    pub max_length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LocalEmbeddingAssetPaths {
    pub onnx_path: PathBuf,
    pub onnx_data_path: Option<PathBuf>,
    pub tokenizer_path: PathBuf,
}

impl Default for LocalEmbeddingModelConfig {
    fn default() -> Self {
        Self::jina()
    }
}

impl LocalEmbeddingModelConfig {
    pub fn jina() -> Self {
        Self {
            source: LocalEmbeddingSource::HuggingFace {
                repo: EMBEDDING_REPO.to_string(),
            },
            onnx_file: EMBEDDING_ONNX_FILE.to_string(),
            onnx_data_file: Some(EMBEDDING_ONNX_DATA_FILE.to_string()),
            tokenizer_file: EMBEDDING_TOKENIZER_FILE.to_string(),
            embedding_dim: EMBEDDING_DIM,
            pooling: LocalEmbeddingPooling::Mean,
            max_length: EMBEDDING_MAX_LENGTH,
            query_prefix: None,
        }
    }

    pub fn coderankembed() -> Self {
        Self {
            source: LocalEmbeddingSource::HuggingFace {
                repo: CODERANK_EMBEDDING_REPO.to_string(),
            },
            onnx_file: EMBEDDING_ONNX_FILE.to_string(),
            onnx_data_file: None,
            tokenizer_file: EMBEDDING_TOKENIZER_FILE.to_string(),
            embedding_dim: EMBEDDING_DIM,
            pooling: LocalEmbeddingPooling::Cls,
            max_length: EMBEDDING_MAX_LENGTH,
            query_prefix: Some(CODERANK_QUERY_PREFIX.to_string()),
        }
    }

    pub fn from_huggingface_repo(repo: impl Into<String>) -> Self {
        let source = LocalEmbeddingSource::HuggingFace { repo: repo.into() };
        let mut defaults = Self::defaults_for_source(&source);
        defaults.source = source;
        defaults
    }

    pub fn from_directory(path: PathBuf) -> Self {
        Self {
            source: LocalEmbeddingSource::Directory { path },
            ..Self::default()
        }
    }

    /// Switch to the FP16 ONNX model when running on a GPU execution provider.
    ///
    /// Quantized INT8 models use operators (QLinearMatMul, MatMulInteger) that
    /// lack CUDA/ROCm/DirectML kernels, so ORT silently falls back to CPU for
    /// those nodes. FP16 runs natively on GPU and is much faster.
    ///
    /// Only applies to the default Jina model; custom user overrides are left
    /// untouched.
    pub fn adjust_for_gpu(&mut self, ep: crate::config::OnnxExecutionProvider) {
        if ep == crate::config::OnnxExecutionProvider::Cpu {
            tracing::debug!("adjust_for_gpu: CPU backend, keeping {}", self.onnx_file);
            return;
        }
        // Only swap if the user hasn't overridden the ONNX file to a
        // non-default value via env vars. Note: the CLI config loader sets
        // this env var from saved config even for default values, so we
        // check the actual value, not just presence.
        if let Some(env_val) = env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV) {
            if env_val != EMBEDDING_ONNX_FILE {
                tracing::debug!(
                    "adjust_for_gpu: user overrode ONNX file via env to {env_val}, skipping swap"
                );
                return;
            }
        }
        if self.onnx_file == EMBEDDING_ONNX_FILE {
            tracing::info!(
                "GPU backend ({ep}): switching from quantized to fp16 model (INT8 ops lack GPU kernels)"
            );
            self.onnx_file = EMBEDDING_ONNX_GPU_FILE.to_string();
            self.onnx_data_file = Some(EMBEDDING_ONNX_GPU_DATA_FILE.to_string());
        } else {
            tracing::debug!(
                "adjust_for_gpu: onnx_file={} is not default quantized, no swap needed",
                self.onnx_file
            );
        }
    }

    pub fn from_env() -> Result<Self> {
        let repo = env_override(LOCAL_EMBEDDING_REPO_ENV);
        let dir = env_override(LOCAL_EMBEDDING_DIR_ENV);

        let source = match (repo, dir) {
            (Some(repo), None) => {
                return Self::apply_env_overrides(Self::from_huggingface_repo(
                    normalize_huggingface_repo(&repo)?,
                ));
            }
            (None, Some(path)) => {
                return Self::apply_env_overrides(Self::from_directory(PathBuf::from(path)));
            }
            (None, None) => Self::default().source,
            (Some(_), Some(_)) => anyhow::bail!(
                "{LOCAL_EMBEDDING_REPO_ENV} and {LOCAL_EMBEDDING_DIR_ENV} cannot both be set"
            ),
        };
        Self::apply_env_overrides(Self::defaults_for_source(&source))
    }

    pub fn display_name(&self) -> String {
        match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => repo.clone(),
            LocalEmbeddingSource::Directory { path } => path.display().to_string(),
        }
    }

    pub fn model_identity(&self) -> String {
        if self == &Self::jina() || self == &Self::coderankembed() {
            return self.display_name();
        }

        let source = match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => format!("hf:{repo}"),
            LocalEmbeddingSource::Directory { path } => format!("dir:{}", path.display()),
        };
        let onnx_data = self.onnx_data_file.as_deref().unwrap_or("-");
        format!(
            "{source}|onnx={}|onnx_data={onnx_data}|tokenizer={}|pooling={}|dim={}|max_length={}",
            self.onnx_file, self.tokenizer_file, self.pooling, self.embedding_dim, self.max_length
        )
    }

    pub fn query_text(&self, query: &str) -> String {
        let Some(prefix) = self
            .query_prefix
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return query.to_string();
        };

        if prefix.chars().last().is_some_and(char::is_whitespace) {
            format!("{prefix}{query}")
        } else {
            format!("{prefix} {query}")
        }
    }

    pub fn cached_asset_paths(&self) -> Result<LocalEmbeddingAssetPaths> {
        let base_dir = match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => {
                vera_home_dir()?.join("models").join(repo)
            }
            LocalEmbeddingSource::Directory { path } => path.clone(),
        };
        Ok(LocalEmbeddingAssetPaths {
            onnx_path: base_dir.join(&self.onnx_file),
            onnx_data_path: self.onnx_data_file.as_ref().map(|path| base_dir.join(path)),
            tokenizer_path: base_dir.join(&self.tokenizer_file),
        })
    }

    fn defaults_for_source(source: &LocalEmbeddingSource) -> Self {
        match source {
            LocalEmbeddingSource::HuggingFace { repo } if repo == CODERANK_EMBEDDING_REPO => {
                Self::coderankembed()
            }
            _ => Self::default(),
        }
    }

    fn apply_env_overrides(defaults: Self) -> Result<Self> {
        Ok(Self {
            source: defaults.source,
            onnx_file: env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV)
                .unwrap_or_else(|| defaults.onnx_file.clone()),
            onnx_data_file: env_optional_override(
                LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
                defaults.onnx_data_file.clone(),
            ),
            tokenizer_file: env_override(LOCAL_EMBEDDING_TOKENIZER_FILE_ENV)
                .unwrap_or_else(|| defaults.tokenizer_file.clone()),
            embedding_dim: parse_env_usize(LOCAL_EMBEDDING_DIM_ENV, defaults.embedding_dim)?,
            pooling: parse_pooling_env(LOCAL_EMBEDDING_POOLING_ENV, defaults.pooling)?,
            max_length: parse_env_usize(LOCAL_EMBEDDING_MAX_LENGTH_ENV, defaults.max_length)?,
            query_prefix: parse_query_prefix_from_env().or_else(|| defaults.query_prefix.clone()),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalModelAssetStatus {
    pub name: &'static str,
    pub path: PathBuf,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedLibraryDependencyStatus {
    pub inspected_files: Vec<PathBuf>,
    pub missing_details: Vec<String>,
    pub missing_libraries: Vec<String>,
}

fn default_embedding_max_length() -> usize {
    EMBEDDING_MAX_LENGTH
}

fn env_override(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_optional_override(key: &str, default: Option<String>) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if value.trim().is_empty() => None,
        Ok(value) => Some(value.trim().to_string()),
        Err(_) => default,
    }
}

fn parse_env_usize(key: &str, default: usize) -> Result<usize> {
    match env_override(key) {
        Some(value) => value
            .parse::<usize>()
            .with_context(|| format!("invalid {key}: {value}")),
        None => Ok(default),
    }
}

fn parse_pooling_env(key: &str, default: LocalEmbeddingPooling) -> Result<LocalEmbeddingPooling> {
    match env_override(key) {
        Some(value) => value
            .parse::<LocalEmbeddingPooling>()
            .map_err(anyhow::Error::msg),
        None => Ok(default),
    }
}

fn parse_query_prefix_from_env() -> Option<String> {
    env_override(LOCAL_EMBEDDING_QUERY_PREFIX_ENV)
        .or_else(|| env_override(LEGACY_EMBEDDING_QUERY_PREFIX_ENV))
}

pub fn normalize_huggingface_repo(value: &str) -> Result<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        anyhow::bail!("embedding repo cannot be empty");
    }

    if let Some(rest) = trimmed
        .strip_prefix("https://huggingface.co/")
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
    {
        let mut parts = rest.split('/').filter(|part| !part.is_empty());
        let owner = parts
            .next()
            .context("invalid Hugging Face URL: missing owner")?;
        let repo = parts
            .next()
            .context("invalid Hugging Face URL: missing repo")?;
        return Ok(format!("{owner}/{repo}"));
    }

    if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        anyhow::bail!("unsupported embedding repo URL: {trimmed}");
    }

    Ok(trimmed.to_string())
}

/// Ensure the ONNX Runtime shared library is loaded and initialized.
///
/// Accepts an optional pre-resolved library path (from `ensure_ort_library`).
/// Falls back to system library search if no path is provided.
///
/// Safe to call multiple times — only the first call takes effect.
pub fn ensure_ort_runtime(lib_path: Option<&std::path::Path>) -> Result<()> {
    let result = ORT_INIT_RESULT.get_or_init(|| {
        let lib_name = match lib_path {
            Some(p) => p.display().to_string(),
            None => ort_lib_filename(),
        };
        match ort::init_from(&lib_name) {
            Ok(builder) => {
                builder.commit();
                Ok(())
            }
            Err(e) => Err(format!(
                "ONNX Runtime shared library not found.\n\
                 Run `vera setup` to auto-download it, or use API mode instead.\n\
                 Original error: {e}"
            )),
        }
    });

    match result {
        Ok(()) => Ok(()),
        Err(msg) => anyhow::bail!("{msg}"),
    }
}

/// Return Vera's home directory.
///
/// Resolution order:
/// 1. `VERA_HOME` env var (explicit override)
/// 2. `~/.vera` if it already exists (backward compatibility)
/// 3. `$XDG_DATA_HOME/vera` (XDG standard, defaults to `~/.local/share/vera`)
/// 4. `~/.vera` as final fallback
pub fn vera_home_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("VERA_HOME") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let home = dirs::home_dir().context("Could not find home directory")?;
    let legacy = home.join(".vera");
    if legacy.exists() {
        return Ok(legacy);
    }

    if let Some(data) = dirs::data_dir() {
        return Ok(data.join("vera"));
    }

    Ok(legacy)
}

/// Get the platform-specific ONNX Runtime shared library filename.
fn ort_lib_filename() -> String {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return path;
        }
    }

    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll".to_string()
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so".to_string()
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib".to_string()
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        "libonnxruntime.so".to_string()
    }
}

use crate::config::OnnxExecutionProvider;

fn parse_cuda_major_version(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let normalized = trimmed.trim_matches(|ch: char| ch == '"' || ch.is_whitespace());
    let last_segment = normalized
        .rsplit(['\\', '/'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(normalized);
    let version_segment = last_segment
        .strip_prefix('v')
        .or_else(|| last_segment.strip_prefix("cuda-"))
        .unwrap_or(last_segment);
    [version_segment, last_segment, normalized]
        .into_iter()
        .find_map(parse_cuda_major_version_tokens)
}

fn parse_cuda_major_version_tokens(value: &str) -> Option<u32> {
    value
        .split_whitespace()
        .find_map(parse_cuda_major_version_token)
}

fn parse_cuda_major_version_token(value: &str) -> Option<u32> {
    let token = value.trim_matches(|ch: char| ch == '"' || ch == ',' || ch == ':' || ch == '=');
    let version_token = token
        .strip_prefix('v')
        .or_else(|| token.strip_prefix("cuda-"))
        .unwrap_or(token);
    version_token
        .split(['.', '_', '-'])
        .next()
        .and_then(|major| major.parse::<u32>().ok())
}

fn detect_cuda_major_from_cuda_path_value(value: &str) -> Option<u32> {
    parse_cuda_major_version(value).or_else(|| {
        let cuda_root = Path::new(value);
        detect_cuda_major_from_cuda_version_file(&cuda_root.join("version.json"))
            .or_else(|| detect_cuda_major_from_cuda_version_file(&cuda_root.join("version.txt")))
    })
}

fn detect_cuda_major_from_cuda_version_file(path: &Path) -> Option<u32> {
    let contents = std::fs::read_to_string(path).ok()?;
    parse_cuda_major_from_cuda_version_metadata(&contents)
}

fn parse_cuda_major_from_cuda_version_metadata(value: &str) -> Option<u32> {
    parse_cuda_major_from_cuda_version_json(value)
        .or_else(|| value.lines().find_map(parse_cuda_major_version))
}

fn parse_cuda_major_from_cuda_version_json(value: &str) -> Option<u32> {
    fn find_cuda_version(value: &serde_json::Value) -> Option<u32> {
        match value {
            serde_json::Value::Object(map) => map
                .get("cuda")
                .and_then(find_cuda_version)
                .or_else(|| {
                    map.get("version")
                        .and_then(|version| version.as_str())
                        .and_then(parse_cuda_major_version)
                })
                .or_else(|| map.values().find_map(find_cuda_version)),
            serde_json::Value::Array(values) => values.iter().find_map(find_cuda_version),
            _ => None,
        }
    }

    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|json| find_cuda_version(&json))
}

fn detect_cuda_major_from_cuda_path_env_vars<T, U>(
    vars: impl IntoIterator<Item = (T, U)>,
) -> Option<u32>
where
    T: AsRef<str>,
    U: AsRef<str>,
{
    vars.into_iter().find_map(|(key, value)| {
        key.as_ref().strip_prefix("CUDA_PATH_V").and_then(|suffix| {
            parse_cuda_major_version(suffix)
                .or_else(|| detect_cuda_major_from_cuda_path_value(value.as_ref()))
        })
    })
}

fn effective_cuda_major(detected_cuda_major: Option<u32>) -> u32 {
    detected_cuda_major.unwrap_or(DEFAULT_CUDA_MAJOR)
}

fn uses_cuda13_ort(detected_cuda_major: Option<u32>) -> bool {
    effective_cuda_major(detected_cuda_major) >= CUDA_13_ORT_MIN_MAJOR
}

fn cuda_ort_cache_dir_name(detected_cuda_major: Option<u32>) -> &'static str {
    if uses_cuda13_ort(detected_cuda_major) {
        "cuda13"
    } else {
        "cuda"
    }
}

fn parse_cuda_major_from_runtime_library_entry(value: &str) -> Option<u32> {
    CUDA_RUNTIME_LIBRARY_PREFIXES
        .iter()
        .filter_map(|prefix| {
            value.find(prefix).and_then(|start| {
                let digits: String = value[start + prefix.len()..]
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect();
                (!digits.is_empty())
                    .then(|| digits.parse::<u32>().ok())
                    .flatten()
            })
        })
        .max()
}

#[cfg(test)]
fn detect_cuda_major_from_library_entries<T>(entries: impl IntoIterator<Item = T>) -> Option<u32>
where
    T: AsRef<str>,
{
    entries
        .into_iter()
        .filter_map(|entry| parse_cuda_major_from_runtime_library_entry(entry.as_ref()))
        .max()
}

#[cfg(target_os = "linux")]
fn detect_cuda_major_from_library_dirs<T>(dirs: impl IntoIterator<Item = T>) -> Option<u32>
where
    T: AsRef<Path>,
{
    dirs.into_iter()
        .filter_map(|dir| std::fs::read_dir(dir.as_ref()).ok())
        .flat_map(|entries| entries.filter_map(std::result::Result::ok))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| parse_cuda_major_from_runtime_library_entry(&name))
        .max()
}

#[cfg(target_os = "linux")]
fn detect_cuda_major_from_library_dir_groups(groups: &[Vec<PathBuf>]) -> Option<u32> {
    groups
        .iter()
        .find_map(|dirs| detect_cuda_major_from_library_dirs(dirs.iter()))
}

#[cfg(target_os = "linux")]
fn push_unique_library_dir(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
    if !dirs.iter().any(|existing| existing == &dir) {
        dirs.push(dir);
    }
}

#[cfg(target_os = "linux")]
fn cuda_library_dirs_from_cuda_path() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let base = PathBuf::from(cuda_path);
        push_unique_library_dir(&mut dirs, base.join("lib64"));
        push_unique_library_dir(&mut dirs, base.join("targets/x86_64-linux/lib"));
    }
    dirs
}

#[cfg(target_os = "linux")]
fn cuda_library_dirs_from_ld_library_path() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        for path in std::env::split_paths(&paths) {
            push_unique_library_dir(&mut dirs, path);
        }
    }
    dirs
}

#[cfg(target_os = "linux")]
fn default_cuda_library_dirs() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/opt/cuda/lib64"),
        PathBuf::from("/opt/cuda/targets/x86_64-linux/lib"),
        PathBuf::from("/usr/local/cuda/lib64"),
        PathBuf::from("/usr/local/cuda/targets/x86_64-linux/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
    ]
}

fn detect_cuda_major_from_cuda_path() -> Option<u32> {
    std::env::var("CUDA_PATH")
        .ok()
        .and_then(|value| detect_cuda_major_from_cuda_path_value(&value))
        .or_else(|| detect_cuda_major_from_cuda_path_env_vars(std::env::vars()))
}

fn detect_cuda_major_from_nvcc() -> Option<u32> {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let release = stdout.split("release ").nth(1)?;
    release
        .split([',', '\n', '\r'])
        .next()
        .and_then(parse_cuda_major_version)
}

fn detect_cuda_major_from_nvidia_smi() -> Option<u32> {
    let output = std::process::Command::new("nvidia-smi").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let rest = stdout.split("CUDA Version:").nth(1)?;
    rest.split_whitespace()
        .next()
        .and_then(parse_cuda_major_version)
}

#[cfg(target_os = "linux")]
fn detect_cuda_major_from_runtime_libraries() -> Option<u32> {
    let search_groups = [
        cuda_library_dirs_from_cuda_path(),
        cuda_library_dirs_from_ld_library_path(),
    ];
    detect_cuda_major_from_library_dir_groups(&search_groups)
        .or_else(detect_cuda_major_from_ldconfig)
        .or_else(|| detect_cuda_major_from_library_dirs(default_cuda_library_dirs()))
}

#[cfg(not(target_os = "linux"))]
fn detect_cuda_major_from_runtime_libraries() -> Option<u32> {
    None
}

#[cfg(target_os = "linux")]
fn detect_cuda_major_from_ldconfig() -> Option<u32> {
    if !command_exists("ldconfig", &["-p"]) {
        return None;
    }
    let output = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    detect_cuda_major_from_ldconfig_entries(stdout.lines())
}

#[cfg(target_os = "linux")]
fn detect_cuda_major_from_ldconfig_entries<T>(entries: impl IntoIterator<Item = T>) -> Option<u32>
where
    T: AsRef<str>,
{
    entries
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.as_ref();
            ldconfig_entry_matches_host_arch(entry)
                .then(|| parse_cuda_major_from_runtime_library_entry(entry))
                .flatten()
        })
        .max()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn ldconfig_entry_matches_host_arch(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    if [
        "x86-64",
        "x86_64",
        "/lib64/",
        "/usr/lib64/",
        "/x86_64-linux-gnu/",
        "/targets/x86_64-linux/",
    ]
    .iter()
    .any(|marker| value.contains(marker))
    {
        return true;
    }

    ![
        "aarch64",
        "arm64",
        "armhf",
        "armv7",
        "armv8",
        "i386",
        "i486",
        "i586",
        "i686",
        "ppc64",
        "s390x",
        "riscv",
        "/lib32/",
        "/usr/lib32/",
        "/aarch64-linux-gnu/",
        "/arm-linux-gnueabihf/",
        "/i386-linux-gnu/",
        "/i686-linux-gnu/",
        "/ppc64le-linux-gnu/",
        "/s390x-linux-gnu/",
        "/riscv64-linux-gnu/",
    ]
    .iter()
    .any(|marker| value.contains(marker))
}

#[cfg(all(target_os = "linux", not(target_arch = "x86_64")))]
fn ldconfig_entry_matches_host_arch(_: &str) -> bool {
    true
}

/// Detect the CUDA toolkit major version. Prefer the installed toolkit over
/// the driver's maximum supported version so Vera picks the matching ORT build.
fn detect_cuda_major_version() -> Option<u32> {
    detect_cuda_major_from_cuda_path()
        .or_else(detect_cuda_major_from_nvcc)
        .or_else(detect_cuda_major_from_runtime_libraries)
        .or_else(detect_cuda_major_from_nvidia_smi)
}

fn detected_cuda_major_for_ep(ep: OnnxExecutionProvider) -> Option<u32> {
    matches!(ep, OnnxExecutionProvider::Cuda)
        .then(detect_cuda_major_version)
        .flatten()
}

/// Platform-specific ORT archive info: (archive_ext, archive_name, primary_lib_path_inside_archive, local_lib_name, version).
fn ort_platform_info_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<(&'static str, String, String, &'static str, &'static str)> {
    let gpu_suffix = match ep {
        OnnxExecutionProvider::Cpu => "",
        OnnxExecutionProvider::Cuda => {
            let cuda_major = effective_cuda_major(detected_cuda_major);
            if uses_cuda13_ort(detected_cuda_major) {
                tracing::info!("detected CUDA {cuda_major}, using CUDA 13 ORT build");
                "-gpu_cuda13"
            } else {
                tracing::info!("detected CUDA {cuda_major}, using CUDA 12 ORT build");
                "-gpu"
            }
        }
        OnnxExecutionProvider::Rocm => "-rocm",
        OnnxExecutionProvider::DirectMl => "-directml",
        OnnxExecutionProvider::CoreMl => "",
        OnnxExecutionProvider::OpenVino => "-openvino",
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if matches!(ep, OnnxExecutionProvider::DirectMl) {
            anyhow::bail!("DirectML is only supported on Windows");
        }
        if matches!(
            ep,
            OnnxExecutionProvider::OpenVino | OnnxExecutionProvider::Rocm
        ) {
            // These EPs are installed via pip wheels, not GitHub release archives.
            // Return a dummy value; `ensure_ort_library_for_ep` handles them separately.
            let base = format!("onnxruntime-linux-x64{gpu_suffix}-{ORT_VERSION}");
            return Ok((
                "tgz",
                base.clone(),
                format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"),
                "libonnxruntime.so",
                ORT_VERSION,
            ));
        }
        // The CUDA 13 archive is named with `_cuda13` in the filename, but the
        // internal directory inside the tgz always uses plain `-gpu` (no `_cuda13`).
        let archive_name = format!("onnxruntime-linux-x64{gpu_suffix}-{ORT_VERSION}");
        let internal_gpu_suffix = if matches!(ep, OnnxExecutionProvider::Cuda) {
            "-gpu"
        } else {
            gpu_suffix
        };
        let internal_base = format!("onnxruntime-linux-x64{internal_gpu_suffix}-{ORT_VERSION}");
        Ok((
            "tgz",
            archive_name,
            format!("{internal_base}/lib/libonnxruntime.so.{ORT_VERSION}"),
            "libonnxruntime.so",
            ORT_VERSION,
        ))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        if !matches!(ep, OnnxExecutionProvider::Cpu) {
            anyhow::bail!("Only CPU execution provider is supported on Linux aarch64");
        }
        let base = format!("onnxruntime-linux-aarch64-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"),
            "libonnxruntime.so",
            ORT_VERSION,
        ))
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !matches!(
            ep,
            OnnxExecutionProvider::Cpu | OnnxExecutionProvider::CoreMl
        ) {
            anyhow::bail!("Only CPU and CoreML execution providers are supported on macOS ARM");
        }
        let base = format!("onnxruntime-osx-arm64-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.{ORT_VERSION}.dylib"),
            "libonnxruntime.dylib",
            ORT_VERSION,
        ))
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        if !matches!(ep, OnnxExecutionProvider::Cpu) {
            anyhow::bail!("Only CPU execution provider is supported on macOS x86_64");
        }
        let ver = ORT_VERSION_MACOS_X86;
        let base = format!("onnxruntime-osx-x86_64-{ver}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.{ver}.dylib"),
            "libonnxruntime.dylib",
            ORT_VERSION_MACOS_X86,
        ))
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        if matches!(
            ep,
            OnnxExecutionProvider::Rocm | OnnxExecutionProvider::OpenVino
        ) {
            anyhow::bail!("ROCm and OpenVINO are only supported on Linux x86_64");
        }
        // The CUDA 13 archive is named with `_cuda13` in the filename, but the
        // internal directory inside the zip always uses plain `-gpu` (no `_cuda13`).
        let archive_name = format!("onnxruntime-win-x64{gpu_suffix}-{ORT_VERSION}");
        // Internal paths inside the zip always use the plain gpu suffix (no _cuda13).
        let internal_gpu_suffix = if matches!(ep, OnnxExecutionProvider::Cuda) {
            "-gpu"
        } else {
            gpu_suffix
        };
        let internal_base = format!("onnxruntime-win-x64{internal_gpu_suffix}-{ORT_VERSION}");
        let entry = format!("{internal_base}/lib/onnxruntime.dll");
        Ok(("zip", archive_name, entry, "onnxruntime.dll", ORT_VERSION))
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    {
        let _ = (ep, gpu_suffix);
        anyhow::bail!(
            "Unsupported platform for automatic ONNX Runtime download. \
             Install ONNX Runtime manually and set ORT_DYLIB_PATH."
        )
    }
}

/// Returns the pip package name for EPs that require pip-based installation, or None
/// for EPs that have pre-built GitHub release archives.
fn pip_package_for_ep(ep: OnnxExecutionProvider) -> Option<&'static str> {
    match ep {
        OnnxExecutionProvider::OpenVino => Some("onnxruntime-openvino"),
        OnnxExecutionProvider::Rocm => Some("onnxruntime-rocm"),
        _ => None,
    }
}

/// Try installing ORT via pip into a managed venv under `~/.vera/venv/`.
/// Returns the lib directory where .so files were copied on success.
#[cfg(target_os = "linux")]
async fn try_pip_install_ort(ep: OnnxExecutionProvider, lib_dir: &std::path::Path) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    let vera_home = vera_home_dir()?;
    let venv_dir = vera_home.join("venv");

    // Find python3
    let python = find_python3()
        .context("python3 not found. Install Python 3.11+ to enable automatic ORT installation.")?;

    eprintln!("Installing {pkg} via pip (this may take a minute)...");

    // Create venv if it doesn't exist
    if !venv_dir.join("bin").join("python3").exists() {
        eprintln!(
            "  Creating virtual environment at {}...",
            venv_dir.display()
        );
        let status = tokio::process::Command::new(&python)
            .args(["-m", "venv", &venv_dir.to_string_lossy()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .status()
            .await
            .context("failed to create venv")?;
        if !status.success() {
            anyhow::bail!(
                "failed to create virtual environment at {}",
                venv_dir.display()
            );
        }
    }

    let venv_pip = venv_dir.join("bin").join("pip");
    let venv_python = venv_dir.join("bin").join("python3");

    // Upgrade pip quietly, then install the package
    let _ = tokio::process::Command::new(&venv_python)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;

    eprintln!("  Running: pip install {pkg}");
    let output = tokio::process::Command::new(&venv_pip)
        .args(["install", pkg])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("failed to run pip install")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pip install {pkg} failed:\n{stderr}");
    }

    // Find and copy .so files from the installed package
    let site_packages = find_site_packages(&venv_dir)?;
    let capi_dir = site_packages.join("onnxruntime").join("capi");
    if !capi_dir.exists() {
        anyhow::bail!(
            "pip install succeeded but onnxruntime/capi/ not found in {}",
            site_packages.display()
        );
    }

    copy_so_files_from_dir(&capi_dir, lib_dir).await?;
    Ok(())
}

/// Try downloading a wheel directly from PyPI and extracting .so files.
#[cfg(target_os = "linux")]
async fn try_wheel_download_ort(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    let pypi_name = pkg.replace('-', "_");

    eprintln!("Trying direct wheel download from PyPI...");

    crate::init_tls();
    let client = Client::new();

    // Query PyPI JSON API for the latest version's wheel URLs
    let api_url = format!("https://pypi.org/pypi/{pkg}/json");
    let resp = client
        .get(&api_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()
        .context("failed to query PyPI")?;
    let body: serde_json::Value = resp.json().await?;

    // Find a manylinux x86_64 wheel
    let urls = body["urls"]
        .as_array()
        .context("unexpected PyPI response format")?;
    let wheel_url = urls
        .iter()
        .filter_map(|entry| {
            let filename = entry["filename"].as_str()?;
            if filename.contains("linux") && filename.contains("x86_64") {
                entry["url"].as_str().map(|u| u.to_string())
            } else {
                None
            }
        })
        .next()
        .context("no compatible Linux x86_64 wheel found on PyPI")?;

    let version = body["info"]["version"].as_str().unwrap_or("unknown");
    eprintln!("  Downloading {pypi_name} v{version} wheel...");
    eprintln!("  {wheel_url}");

    let res = client
        .get(&wheel_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    // Wheels are zip files; extract .so files from onnxruntime/capi/
    let lib_dir_owned = lib_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_wheel_libs(&bytes, &lib_dir_owned)).await??;

    Ok(())
}

/// Extract all shared libraries from `onnxruntime/capi/` inside a wheel (zip).
#[cfg(target_os = "linux")]
fn extract_wheel_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    let mut extracted = 0usize;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let path = entry.name().to_string();
        if !path.starts_with("onnxruntime/capi/") {
            continue;
        }
        let filename = path.rsplit('/').next().unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(&local_name);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
        }
        create_versioned_symlink(dest_dir, filename, &local_name);
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("no shared libraries found in wheel");
    }
    eprintln!("  Extracted {extracted} libraries from wheel");
    Ok(())
}

/// Find a working python3 binary.
#[cfg(target_os = "linux")]
fn find_python3() -> Option<String> {
    for name in ["python3", "python"] {
        if std::process::Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Some(name.to_string());
        }
    }
    None
}

/// Find the site-packages directory inside a venv.
#[cfg(target_os = "linux")]
fn find_site_packages(venv_dir: &std::path::Path) -> Result<PathBuf> {
    let lib_dir = venv_dir.join("lib");
    if !lib_dir.exists() {
        anyhow::bail!("venv lib directory not found");
    }
    for entry in std::fs::read_dir(&lib_dir)? {
        let entry = entry?;
        let sp = entry.path().join("site-packages");
        if sp.exists() {
            return Ok(sp);
        }
    }
    anyhow::bail!("site-packages not found in venv")
}

/// Copy all .so files from a directory to the target lib directory.
#[cfg(target_os = "linux")]
async fn copy_so_files_from_dir(
    src_dir: &std::path::Path,
    dest_dir: &std::path::Path,
) -> Result<()> {
    let mut extracted = 0usize;
    let mut entries = fs::read_dir(src_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(&local_name);
        fs::copy(&path, &dest).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755)).await?;
        }
        create_versioned_symlink(dest_dir, filename, &local_name);
        extracted += 1;
    }
    if extracted == 0 {
        anyhow::bail!("no .so files found in {}", src_dir.display());
    }
    eprintln!("  Copied {extracted} libraries from pip package");
    Ok(())
}

/// Ensure the ONNX Runtime shared library is available locally.
///
/// Returns the path to the library. Downloads it automatically if needed.
/// Respects `ORT_DYLIB_PATH` — if set, skips auto-download.
/// GPU execution providers download a different (larger) ORT build.
///
/// For OpenVINO and ROCm (no pre-built GitHub archives), tries in order:
/// 1. `pip install` into a managed venv at `~/.vera/venv/`
/// 2. Direct wheel download from PyPI
/// 3. Bail with manual instructions
pub async fn ensure_ort_library_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return ensure_ort_library_for_ep_with_cuda_major(ep, None).await;
        }
    }

    let detected_cuda_major = match ep {
        OnnxExecutionProvider::Cuda => detected_cuda_major_for_ep(ep),
        _ => None,
    };

    ensure_ort_library_for_ep_with_cuda_major(ep, detected_cuda_major).await
}

async fn ensure_ort_library_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    let target_path = ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)?;
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(target_path);
        }
    }

    if target_path.exists() {
        return Ok(target_path);
    }

    let lib_dir = target_path
        .parent()
        .context("failed to determine ONNX Runtime directory")?
        .to_path_buf();

    fs::create_dir_all(&lib_dir).await?;

    // OpenVINO and ROCm: pip-based install with fallback chain
    #[cfg(target_os = "linux")]
    if pip_package_for_ep(ep).is_some() {
        return ensure_ort_via_pip_chain(ep, &lib_dir, &target_path).await;
    }

    // DirectML: distributed via NuGet, not GitHub release archives.
    #[cfg(target_os = "windows")]
    if matches!(ep, OnnxExecutionProvider::DirectMl) {
        return ensure_ort_via_nuget_directml(&lib_dir, &target_path).await;
    }

    // Standard path: download from GitHub releases
    let (ext, archive_name, lib_path_in_archive, local_lib_name, ort_version) =
        ort_platform_info_with_cuda_major(ep, detected_cuda_major)?;
    let is_gpu = ep != OnnxExecutionProvider::Cpu;

    let archive_filename = if ext == "tgz" {
        format!("{archive_name}.tgz")
    } else {
        format!("{archive_name}.zip")
    };
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/{archive_filename}"
    );

    eprintln!("Downloading ONNX Runtime v{ort_version} ({ep})...");
    eprintln!("  {url}");

    crate::init_tls();
    let client = Client::new();
    let res = client
        .get(&url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    let lib_dir_clone = lib_dir.clone();
    let lib_path_in_archive_clone = lib_path_in_archive.clone();

    let extract_result = tokio::task::spawn_blocking(move || -> Result<()> {
        if ext == "tgz" {
            if is_gpu {
                extract_tgz_all_libs(&bytes, &lib_dir_clone)
            } else {
                extract_tgz_single(
                    &bytes,
                    &lib_path_in_archive_clone,
                    &lib_dir_clone.join(local_lib_name),
                )
            }
        } else if is_gpu {
            extract_zip_all_libs(&bytes, &archive_name, &lib_dir_clone)
        } else {
            extract_zip(
                &bytes,
                &lib_path_in_archive_clone,
                &lib_dir_clone.join(local_lib_name),
            )
        }
    })
    .await?;

    if let Err(e) = extract_result {
        return Err(e).context("Failed to extract ONNX Runtime from archive");
    }

    eprintln!(
        "ONNX Runtime v{ort_version} installed to {}",
        lib_dir.display()
    );
    Ok(target_path)
}

/// Re-fetch the ONNX Runtime library for the selected execution provider.
///
/// `vera setup` and `vera repair` call this for CUDA so switching between CUDA
/// toolkits refreshes the downloaded ORT build instead of reusing a stale one.
pub async fn refresh_ort_library_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let detected_cuda_major = detected_cuda_major_for_ep(ep);
    let target_path = preferred_ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)?;
    if ep == OnnxExecutionProvider::Cpu {
        if target_path.exists() {
            fs::remove_file(&target_path).await.with_context(|| {
                format!(
                    "failed to remove stale ONNX Runtime at {}",
                    target_path.display()
                )
            })?;
        }
    } else if let Some(dir) = target_path.parent() {
        if dir.exists() {
            fs::remove_dir_all(dir).await.with_context(|| {
                format!(
                    "failed to remove stale ONNX Runtime directory {}",
                    dir.display()
                )
            })?;
        }
    }

    ensure_ort_library_for_ep_with_cuda_major(ep, detected_cuda_major).await
}

/// Download ONNX Runtime DirectML from NuGet.
///
/// DirectML builds are not published as GitHub release archives; they are only
/// distributed via NuGet. The `.nupkg` is a zip containing DLLs at
/// `runtimes/win-x64/native/`.
#[cfg(target_os = "windows")]
async fn ensure_ort_via_nuget_directml(
    lib_dir: &std::path::Path,
    target_path: &std::path::Path,
) -> Result<PathBuf> {
    let nuget_url = format!(
        "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/{ORT_VERSION}/microsoft.ml.onnxruntime.directml.{ORT_VERSION}.nupkg"
    );

    eprintln!("Downloading ONNX Runtime v{ORT_VERSION} (directml) from NuGet...");
    eprintln!("  {nuget_url}");

    crate::init_tls();
    let client = Client::new();
    let res = client
        .get(&nuget_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()
        .context("failed to download DirectML NuGet package")?;
    let bytes = res.bytes().await?;

    let lib_dir_owned = lib_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        extract_nuget_native_dlls(&bytes, "runtimes/win-x64/native/", &lib_dir_owned)
    })
    .await??;

    eprintln!(
        "ONNX Runtime v{ORT_VERSION} (directml) installed to {}",
        lib_dir.display()
    );
    Ok(target_path.to_path_buf())
}

/// Extract native DLLs from a NuGet package (zip) at the given prefix.
#[cfg(target_os = "windows")]
fn extract_nuget_native_dlls(data: &[u8], prefix: &str, dest_dir: &std::path::Path) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    let mut extracted = 0usize;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let path = entry.name().to_string();
        if !path.starts_with(prefix) {
            continue;
        }
        let filename = path.rsplit('/').next().unwrap_or("");
        if !filename.ends_with(".dll") {
            continue;
        }
        let dest = dest_dir.join(filename);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("no DLLs found in NuGet package at {prefix}");
    }
    eprintln!("  Extracted {extracted} libraries from NuGet package");
    Ok(())
}

/// Pip-based fallback chain for OpenVINO and ROCm.
#[cfg(target_os = "linux")]
async fn ensure_ort_via_pip_chain(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
    target_path: &std::path::Path,
) -> Result<PathBuf> {
    let pkg = pip_package_for_ep(ep).unwrap();

    // Option 1: pip install into managed venv
    match try_pip_install_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!(
                "ONNX Runtime ({ep}) installed via pip to {}",
                lib_dir.display()
            );
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("pip install failed, trying direct wheel download: {e:#}");
            eprintln!("  pip install failed, trying direct wheel download...");
        }
    }

    // Option 2: download wheel directly from PyPI
    match try_wheel_download_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!(
                "ONNX Runtime ({ep}) installed via wheel to {}",
                lib_dir.display()
            );
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("wheel download failed: {e:#}");
            eprintln!("  Wheel download also failed.");
        }
    }

    // Option 3: bail with manual instructions
    anyhow::bail!(
        "Could not automatically install ONNX Runtime with {ep} support.\n\
         Install manually:\n\
         \n\
         1. pip install {pkg}\n\
         2. Locate libonnxruntime.so inside the installed package:\n\
            python3 -c \"import onnxruntime; import os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))\"\n\
         3. Set ORT_DYLIB_PATH to the full path of libonnxruntime.so\n\
         4. Run `vera setup` again"
    )
}

pub fn ort_library_path_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let detected_cuda_major = match ep {
        OnnxExecutionProvider::Cuda => detected_cuda_major_for_ep(ep),
        _ => None,
    };

    ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)
}

fn preferred_ort_library_path_for_ep_in_home(
    vera_home: &Path,
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> PathBuf {
    let lib_dir = match ep {
        OnnxExecutionProvider::Cpu => vera_home.join("lib"),
        OnnxExecutionProvider::Cuda => vera_home
            .join("lib")
            .join(cuda_ort_cache_dir_name(detected_cuda_major)),
        _ => vera_home.join("lib").join(ep.to_string()),
    };

    lib_dir.join(platform_ort_lib_name())
}

fn cached_ort_library_path_for_ep_in_home(
    vera_home: &Path,
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> PathBuf {
    let preferred_path =
        preferred_ort_library_path_for_ep_in_home(vera_home, ep, detected_cuda_major);
    if matches!(ep, OnnxExecutionProvider::Cuda)
        && detected_cuda_major.is_none()
        && !preferred_path.exists()
    {
        let cuda13_path =
            preferred_ort_library_path_for_ep_in_home(vera_home, ep, Some(CUDA_13_ORT_MIN_MAJOR));
        if cuda13_path.exists() {
            return cuda13_path;
        }
    }
    preferred_path
}

fn preferred_ort_library_path_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let vera_home = vera_home_dir()?;
    Ok(preferred_ort_library_path_for_ep_in_home(
        &vera_home,
        ep,
        detected_cuda_major,
    ))
}

fn ort_library_path_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let vera_home = vera_home_dir()?;
    Ok(cached_ort_library_path_for_ep_in_home(
        &vera_home,
        ep,
        detected_cuda_major,
    ))
}

pub fn ensure_provider_dependencies(
    ep: OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> Result<()> {
    // CPU mode only needs the core runtime library; skip provider dependency checks.
    if matches!(ep, OnnxExecutionProvider::Cpu) {
        return Ok(());
    }

    let Some(status) = inspect_shared_library_deps(runtime_path)? else {
        return Ok(());
    };

    let mut missing_details = status.missing_details;
    if matches!(ep, OnnxExecutionProvider::Cuda) {
        // CUDA builds ship optional TensorRT provider libraries alongside the
        // CUDA provider. Missing TensorRT does not block plain CUDA execution.
        missing_details.retain(|detail| !detail.starts_with("libonnxruntime_providers_tensorrt"));
    }

    let mut missing_libraries: Vec<String> = missing_details
        .iter()
        .filter_map(|detail| detail.split(": ").nth(1))
        .map(str::to_string)
        .collect();
    missing_libraries.sort();
    missing_libraries.dedup();

    if missing_libraries.is_empty() {
        return Ok(());
    }

    let backend_name = match ep {
        OnnxExecutionProvider::Cpu => "CPU",
        OnnxExecutionProvider::Cuda => "CUDA",
        OnnxExecutionProvider::Rocm => "ROCm",
        OnnxExecutionProvider::DirectMl => "DirectML",
        OnnxExecutionProvider::CoreMl => "CoreML",
        OnnxExecutionProvider::OpenVino => "OpenVINO",
    };

    let mut message =
        format!("{backend_name} backend selected, but required libraries are missing:\n");
    for library in &missing_libraries {
        message.push_str(&format!("  {library}\n"));
    }
    if let Some(hint) = dependency_hint(ep) {
        message.push_str(&hint);
        message.push('\n');
    }
    message.push_str("Run `vera doctor --probe` for details.");
    anyhow::bail!("{}", message.trim_end());
}

pub fn inspect_shared_library_deps(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    inspect_shared_library_deps_impl(runtime_path)
}

#[cfg(target_os = "linux")]
fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("ldd", &["--version"]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".so");

    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let output = std::process::Command::new("ldd")
            .arg(inspected)
            .output()
            .with_context(|| format!("failed to run `ldd` on {}", inspected.display()))?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for line in text.lines().filter(|line| line.contains("not found")) {
            let library = line.split("=>").next().unwrap_or(line).trim().to_string();
            missing_details.push(format!("{file_name}: {library}"));
            missing_libraries.push(library);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "macos")]
fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("otool", &["-L", runtime_path.to_string_lossy().as_ref()]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".dylib");
    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let dependencies = macos_dependencies(inspected)?;
        let rpaths = macos_rpaths(inspected)?;
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for dependency in dependencies {
            if macos_dependency_exists(&dependency, inspected, &rpaths) {
                continue;
            }
            missing_details.push(format!("{file_name}: {dependency}"));
            missing_libraries.push(dependency);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "windows")]
fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".dll");

    // Use dumpbin if available (Visual Studio), otherwise fall back to a
    // known-list check for CUDA provider DLLs alongside onnxruntime.dll.
    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    if command_exists("dumpbin", &["/?"]) {
        for inspected in &inspected_files {
            let output = std::process::Command::new("dumpbin")
                .args(["/dependents", inspected.to_string_lossy().as_ref()])
                .output();
            let Ok(output) = output else { continue };
            let text = String::from_utf8_lossy(&output.stdout);
            let file_name = inspected
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            for line in text.lines() {
                let dep = line.trim();
                if dep.ends_with(".dll") && !dep.contains(' ') {
                    // Check if the DLL can be found by the loader: same dir or system PATH.
                    let in_same_dir = inspected
                        .parent()
                        .map(|dir| dir.join(dep).exists())
                        .unwrap_or(false);
                    if !in_same_dir && which_dll(dep).is_none() {
                        missing_details.push(format!("{file_name}: {dep}"));
                        missing_libraries.push(dep.to_string());
                    }
                }
            }
        }
    } else {
        // No dumpbin: check that expected CUDA provider DLLs exist alongside the runtime.
        if let Some(dir) = runtime_path.parent() {
            let expected_cuda_libs = [
                "onnxruntime_providers_shared.dll",
                "onnxruntime_providers_cuda.dll",
            ];
            for lib in &expected_cuda_libs {
                if !dir.join(lib).exists() {
                    missing_details.push(format!("onnxruntime.dll: {lib}"));
                    missing_libraries.push(lib.to_string());
                }
            }
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "windows")]
fn which_dll(name: &str) -> Option<std::path::PathBuf> {
    let path_var = std::env::var("PATH").unwrap_or_default();
    for dir in path_var.split(';') {
        let candidate = std::path::Path::new(dir).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn inspect_shared_library_deps_impl(
    _: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    Ok(None)
}

fn collect_runtime_libraries(runtime_path: &std::path::Path, suffix: &str) -> Vec<PathBuf> {
    let mut inspected_files = vec![runtime_path.to_path_buf()];
    if let Some(dir) = runtime_path.parent() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if (name.starts_with("libonnxruntime") || name.starts_with("onnxruntime"))
                    && name.contains(suffix)
                    && path != runtime_path
                {
                    inspected_files.push(path);
                }
            }
        }
    }
    inspected_files.sort();
    inspected_files.dedup();
    inspected_files
}

fn command_exists(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

#[cfg(target_os = "macos")]
fn macos_dependencies(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-L", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -L` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

#[cfg(target_os = "macos")]
fn macos_rpaths(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-l", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -l` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut rpaths = Vec::new();
    let mut in_rpath = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "cmd LC_RPATH" {
            in_rpath = true;
            continue;
        }
        if in_rpath && trimmed.starts_with("path ") {
            if let Some(path) = trimmed
                .strip_prefix("path ")
                .and_then(|rest| rest.split(" (offset").next())
            {
                rpaths.push(path.trim().to_string());
            }
            in_rpath = false;
        }
    }
    Ok(rpaths)
}

#[cfg(target_os = "macos")]
fn macos_dependency_exists(
    dependency: &str,
    inspected: &std::path::Path,
    rpaths: &[String],
) -> bool {
    if dependency.starts_with("/System/Library/") || dependency.starts_with("/usr/lib/") {
        return true;
    }

    if let Some(rest) = dependency.strip_prefix("@loader_path/") {
        return inspected
            .parent()
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@executable_path/") {
        let exe_path = std::env::current_exe().ok();
        let exe_exists = exe_path
            .as_deref()
            .and_then(|exe| exe.parent())
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
        return exe_exists
            || inspected
                .parent()
                .map(|parent| parent.join(rest).exists())
                .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@rpath/") {
        return rpaths
            .iter()
            .map(|rpath| resolve_macos_rpath(rpath, inspected))
            .any(|candidate| candidate.join(rest).exists());
    }

    if dependency.starts_with('@') {
        return false;
    }

    std::path::Path::new(dependency).exists()
}

#[cfg(target_os = "macos")]
fn resolve_macos_rpath(rpath: &str, inspected: &std::path::Path) -> PathBuf {
    if rpath == "@loader_path" || rpath.starts_with("@loader_path/") {
        let rest = rpath.strip_prefix("@loader_path/").unwrap_or("");
        return inspected
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .join(rest);
    }

    if rpath == "@executable_path" || rpath.starts_with("@executable_path/") {
        let rest = rpath.strip_prefix("@executable_path/").unwrap_or("");
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                return parent.join(rest);
            }
        }
    }

    PathBuf::from(rpath)
}

fn dependency_hint(ep: OnnxExecutionProvider) -> Option<String> {
    match ep {
        OnnxExecutionProvider::Cpu => None,
        OnnxExecutionProvider::Cuda => Some(match detect_cuda_major_version() {
            Some(cuda_major) => format!(
                "Install the CUDA {cuda_major} toolkit and cuDNN 9, then ensure they're on the linker path."
            ),
            None => {
                "Install the CUDA toolkit and cuDNN 9, then ensure they're on the linker path."
                    .to_string()
            }
        }),
        OnnxExecutionProvider::Rocm => {
            Some("Install the ROCm userspace libraries, then ensure they're on the linker path.".to_string())
        }
        OnnxExecutionProvider::DirectMl => {
            Some("Install the DirectML runtime and required GPU drivers.".to_string())
        }
        OnnxExecutionProvider::CoreMl => {
            Some("Verify you are running on Apple Silicon with a supported macOS version.".to_string())
        }
        OnnxExecutionProvider::OpenVino => {
            Some("Install the Intel OpenVINO runtime or compute libraries, then ensure they're on the linker path.".to_string())
        }
    }
}

fn platform_ort_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Extract a single shared library from a tgz archive (CPU builds).
fn extract_tgz_single(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.to_string_lossy() == entry_path {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    // Suffix fallback (archive structure may vary)
    let decoder2 = GzDecoder::new(data);
    let mut archive2 = tar::Archive::new(decoder2);
    let suffix = entry_path.rsplit('/').next().unwrap_or(entry_path);

    for entry in archive2.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if path_str.ends_with(suffix) && path_str.contains("/lib/") {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    anyhow::bail!("Could not find {entry_path} in ORT archive")
}

/// Extract all shared libraries from a tgz archive (GPU builds need provider libs).
fn extract_tgz_all_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = 0usize;

    for entry in archive.entries()? {
        let mut entry = entry?;
        // Skip symlinks — we want the real files
        if entry.header().entry_type() != tar::EntryType::Regular {
            continue;
        }
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if !path_str.contains("/lib/") {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string();
        // Extract .so, .dylib, .dll files (skip .pc and other non-library files)
        let is_lib =
            filename.contains(".so") || filename.ends_with(".dylib") || filename.ends_with(".dll");
        if !is_lib {
            continue;
        }
        // Normalize versioned names: libonnxruntime.so.1.23.2 → libonnxruntime.so
        let local_name = strip_so_version(&filename);
        let dest = dest_dir.join(&local_name);
        write_lib_file(&mut entry, &dest)?;
        create_versioned_symlink(dest_dir, &filename, &local_name);
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("No shared libraries found in ORT archive");
    }
    Ok(())
}

/// Strip .so version suffix: "libonnxruntime.so.1.23.2" → "libonnxruntime.so"
fn strip_so_version(name: &str) -> String {
    if let Some(pos) = name.find(".so.") {
        name[..pos + 3].to_string()
    } else {
        name.to_string()
    }
}

/// Create a versioned symlink if the original filename differs from the
/// stripped name. For example, if `original` is "libopenvino.so.2541" and
/// `stripped` is "libopenvino.so", creates a symlink
/// `dest_dir/libopenvino.so.2541 -> libopenvino.so`.
#[cfg(unix)]
fn create_versioned_symlink(dest_dir: &std::path::Path, original: &str, stripped: &str) {
    if original == stripped {
        return;
    }
    let link = dest_dir.join(original);
    if link.exists() || link.symlink_metadata().is_ok() {
        return;
    }
    if let Err(e) = std::os::unix::fs::symlink(stripped, &link) {
        tracing::debug!(link = %link.display(), target = stripped, error = %e, "failed to create versioned symlink");
    }
}

#[cfg(not(unix))]
fn create_versioned_symlink(_dest_dir: &std::path::Path, _original: &str, _stripped: &str) {}

fn write_lib_file(reader: &mut impl std::io::Read, dest: &std::path::Path) -> Result<()> {
    let mut out = std::fs::File::create(dest)?;
    std::io::copy(reader, &mut out)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

fn extract_zip(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    // Windows .zip extraction using tar crate's zip support is not available,
    // so we use a minimal zip reader via the `zip` crate. Since we only compile
    // this path on Windows and want to avoid an extra dependency, we fall back
    // to extracting via the system `tar` command or manual parsing.
    //
    // For simplicity, use the zip crate. But since it's not added as a dep,
    // we'll use a raw approach: download the tgz variant if available, or
    // shell out to PowerShell on Windows.
    #[cfg(target_os = "windows")]
    {
        let temp_zip = dest.with_extension("zip");
        std::fs::write(&temp_zip, data)?;
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                     $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                     $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                     if ($entry) {{ \
                         $stream = $entry.Open(); \
                         $file = [System.IO.File]::Create('{}'); \
                         $stream.CopyTo($file); \
                         $file.Close(); $stream.Close(); \
                     }}; $zip.Dispose()",
                    temp_zip.display(),
                    entry_path.replace('/', "\\"),
                    dest.display()
                ),
            ])
            .output()?;
        let _ = std::fs::remove_file(&temp_zip);
        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract ORT zip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if !dest.exists() {
            // Try with forward slashes
            let temp_zip2 = dest.with_extension("zip2");
            std::fs::write(&temp_zip2, data)?;
            let output2 = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                         $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                         $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                         if ($entry) {{ \
                             $stream = $entry.Open(); \
                             $file = [System.IO.File]::Create('{}'); \
                             $stream.CopyTo($file); \
                             $file.Close(); $stream.Close(); \
                         }}; $zip.Dispose()",
                        temp_zip2.display(),
                        entry_path,
                        dest.display()
                    ),
                ])
                .output()?;
            let _ = std::fs::remove_file(&temp_zip2);
            if !output2.status.success() || !dest.exists() {
                anyhow::bail!("Could not find {entry_path} in ORT zip archive");
            }
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (data, entry_path, dest);
        anyhow::bail!("ZIP extraction not expected on this platform")
    }
}

/// Extract all DLL files from a zip archive's lib/ directory (GPU builds need provider libs).
fn extract_zip_all_libs(
    data: &[u8],
    _archive_name: &str,
    dest_dir: &std::path::Path,
) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let temp_zip = dest_dir.join("_ort_download.zip");
        std::fs::write(&temp_zip, data)?;
        // Extract all .dll files from any lib/ subdirectory inside the archive.
        // We match loosely because the CUDA 13 archive filename contains `_cuda13`
        // but the internal directory does not.
        let script = format!(
            "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
             $zip = [System.IO.Compression.ZipFile]::OpenRead('{zip}'); \
             $count = 0; \
             foreach ($entry in $zip.Entries) {{ \
                 if ($entry.FullName -match '/lib/[^/]+\\.dll$' -and $entry.Length -gt 0) {{ \
                     $name = $entry.Name; \
                     $dest = Join-Path '{dest}' $name; \
                     $stream = $entry.Open(); \
                     $file = [System.IO.File]::Create($dest); \
                     $stream.CopyTo($file); \
                     $file.Close(); $stream.Close(); \
                     $count++; \
                 }} \
             }}; \
             $zip.Dispose(); \
             if ($count -eq 0) {{ exit 1 }}",
            zip = temp_zip.display(),
            dest = dest_dir.display(),
        );
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()?;
        let _ = std::fs::remove_file(&temp_zip);
        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract DLLs from ORT zip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (data, _archive_name, dest_dir);
        anyhow::bail!("ZIP extraction not expected on this platform")
    }
}

/// Wrap an ort error with a user-friendly message suggesting alternatives.
pub fn wrap_ort_error(e: impl std::fmt::Display) -> String {
    let err_msg = e.to_string();
    let lower = err_msg.to_ascii_lowercase();
    if lower.contains("required libraries are missing") {
        return err_msg;
    }
    if lower.contains("libonnxruntime")
        || lower.contains("onnxruntime.dll")
        || lower.contains("libonnxruntime.so")
        || lower.contains("libonnxruntime.dylib")
        || lower.contains("loadlibrary")
        || lower.contains("shared library")
        || lower.contains("specified module could not be found")
        || lower.contains(".dll")
        || lower.contains(".dylib")
        || lower.contains(".so")
    {
        format!(
            "ONNX Runtime shared library not found.\n\
             Run `vera setup` to auto-download it, or use API mode instead.\n\
             Original error: {err_msg}"
        )
    } else {
        format!(
            "Failed to initialize ONNX session: {err_msg}\nRun `vera doctor --probe` for details."
        )
    }
}

/// Download a file from HuggingFace Hub using atomic writes.
pub async fn ensure_model_file(repo_id: &str, file_path: &str) -> Result<PathBuf> {
    ensure_model_file_impl(repo_id, file_path, HUB_URL, None).await
}

pub fn configured_local_model_name() -> String {
    LocalEmbeddingModelConfig::from_env()
        .map(|config| config.model_identity())
        .unwrap_or_else(|_| EMBEDDING_REPO.to_string())
}

pub async fn ensure_local_embedding_assets(
    config: &LocalEmbeddingModelConfig,
) -> Result<LocalEmbeddingAssetPaths> {
    match &config.source {
        LocalEmbeddingSource::HuggingFace { repo } => Ok(LocalEmbeddingAssetPaths {
            onnx_path: ensure_model_file(repo, &config.onnx_file).await?,
            onnx_data_path: match config.onnx_data_file.as_deref() {
                Some(path) => Some(ensure_model_file(repo, path).await?),
                None => None,
            },
            tokenizer_path: ensure_model_file(repo, &config.tokenizer_file).await?,
        }),
        LocalEmbeddingSource::Directory { .. } => verify_local_embedding_assets(config),
    }
}

fn verify_local_embedding_assets(
    config: &LocalEmbeddingModelConfig,
) -> Result<LocalEmbeddingAssetPaths> {
    let paths = config.cached_asset_paths()?;
    require_existing_file(&paths.onnx_path, "embedding ONNX model")?;
    if let Some(path) = paths.onnx_data_path.as_ref() {
        require_existing_file(path, "embedding ONNX external data")?;
    }
    require_existing_file(&paths.tokenizer_path, "embedding tokenizer")?;
    Ok(paths)
}

fn require_existing_file(path: &Path, label: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    anyhow::bail!(
        "{label} not found at {}.\nHint: place the file there, or point `vera setup` at a Hugging Face repo instead.",
        path.display()
    );
}

/// Download or validate the local embedding model, the curated local reranker, and the ORT library.
pub async fn prepare_local_models_for_ep(
    ep: OnnxExecutionProvider,
    embedding_model: &LocalEmbeddingModelConfig,
) -> Result<Vec<PathBuf>> {
    let mut model = embedding_model.clone();
    model.adjust_for_gpu(ep);
    let mut paths = Vec::new();
    let ort_path = if ep == OnnxExecutionProvider::Cuda {
        refresh_ort_library_for_ep(ep).await?
    } else {
        ensure_ort_library_for_ep(ep).await?
    };
    paths.push(ort_path);
    let embedding_paths = ensure_local_embedding_assets(&model).await?;
    paths.push(embedding_paths.onnx_path);
    if let Some(path) = embedding_paths.onnx_data_path {
        paths.push(path);
    }
    paths.push(embedding_paths.tokenizer_path);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_ONNX_FILE).await?);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_TOKENIZER_FILE).await?);
    Ok(paths)
}

pub fn inspect_local_model_files_for_ep(
    ep: OnnxExecutionProvider,
    embedding_model: &LocalEmbeddingModelConfig,
) -> Result<Vec<LocalModelAssetStatus>> {
    let mut model = embedding_model.clone();
    model.adjust_for_gpu(ep);
    let embedding_paths = model.cached_asset_paths()?;
    let ort_path = ort_library_path_for_ep(ep)?;
    let vera_home = vera_home_dir()?;
    let reranker_onnx = vera_home
        .join("models")
        .join(RERANKER_REPO)
        .join(RERANKER_ONNX_FILE);
    let reranker_tokenizer = vera_home
        .join("models")
        .join(RERANKER_REPO)
        .join(RERANKER_TOKENIZER_FILE);
    let mut assets = vec![
        LocalModelAssetStatus {
            name: "onnx-runtime",
            exists: ort_path.exists(),
            path: ort_path,
        },
        LocalModelAssetStatus {
            name: "embedding-onnx",
            exists: embedding_paths.onnx_path.exists(),
            path: embedding_paths.onnx_path,
        },
        LocalModelAssetStatus {
            name: "embedding-tokenizer",
            exists: embedding_paths.tokenizer_path.exists(),
            path: embedding_paths.tokenizer_path,
        },
        LocalModelAssetStatus {
            name: "reranker-onnx",
            exists: reranker_onnx.exists(),
            path: reranker_onnx,
        },
        LocalModelAssetStatus {
            name: "reranker-tokenizer",
            exists: reranker_tokenizer.exists(),
            path: reranker_tokenizer,
        },
    ];

    if let Some(path) = embedding_paths.onnx_data_path {
        assets.insert(
            2,
            LocalModelAssetStatus {
                name: "embedding-onnx-data",
                exists: path.exists(),
                path,
            },
        );
    }

    Ok(assets)
}

async fn ensure_model_file_impl(
    repo_id: &str,
    file_path: &str,
    base_url: &str,
    home_override: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let home_dir = match home_override {
        Some(p) => p.to_path_buf(),
        None => vera_home_dir()?,
    };
    let models_dir = home_dir.join("models").join(repo_id);
    let target_path = models_dir.join(file_path);

    if target_path.exists() {
        return Ok(target_path);
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let url = format!("{}/{}/resolve/main/{}", base_url, repo_id, file_path);
    eprintln!("Downloading {}...", url);

    crate::init_tls();
    let client = Client::new();
    let res = client.get(&url).send().await?.error_for_status()?;
    let total_size = res.content_length();

    let temp_path = target_path.with_extension(format!("part.{}", std::process::id()));
    let mut file = File::create(&temp_path).await?;
    let mut stream = res.bytes_stream();
    let mut downloaded = 0;

    let download_result: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Download error: {}", e))?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(total) = total_size {
                eprint!(
                    "\rProgress: {} MB / {} MB",
                    downloaded / 1_000_000,
                    total / 1_000_000
                );
            } else {
                eprint!("\rProgress: {} MB", downloaded / 1_000_000);
            }
        }
        file.flush().await?;
        file.sync_all().await?;
        eprintln!("\nDownload complete: {}", file_path);

        if let Err(e) = fs::rename(&temp_path, &target_path).await {
            if target_path.exists() {
                // Another process won the race
                let _ = fs::remove_file(&temp_path).await;
            } else {
                return Err(e.into());
            }
        }
        Ok(())
    }
    .await;

    if let Err(e) = download_result {
        let _ = fs::remove_file(&temp_path).await;
        return Err(e).context(format!(
            "Expected path: {}. Hint: check network connection or manually place model at {}",
            target_path.display(),
            target_path.display()
        ));
    }

    Ok(target_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn normalize_huggingface_repo_accepts_repo_ids_and_urls() {
        assert_eq!(
            normalize_huggingface_repo("Zenabius/CodeRankEmbed-onnx").unwrap(),
            "Zenabius/CodeRankEmbed-onnx"
        );
        assert_eq!(
            normalize_huggingface_repo("https://huggingface.co/Zenabius/CodeRankEmbed-onnx")
                .unwrap(),
            "Zenabius/CodeRankEmbed-onnx"
        );
    }

    #[test]
    fn coderankembed_preset_sets_required_query_prefix() {
        let config = LocalEmbeddingModelConfig::coderankembed();
        assert_eq!(
            config.query_text("find router code"),
            "Represent this query for searching relevant code: find router code"
        );
    }

    #[test]
    fn coderankembed_repo_uses_coderank_defaults() {
        let config =
            LocalEmbeddingModelConfig::from_huggingface_repo("Zenabius/CodeRankEmbed-onnx");
        assert_eq!(config.pooling, LocalEmbeddingPooling::Cls);
        assert!(config.onnx_data_file.is_none());
    }

    #[test]
    fn parse_cuda_major_version_handles_paths_and_versions() {
        assert_eq!(
            parse_cuda_major_version(r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"#),
            Some(13)
        );
        assert_eq!(parse_cuda_major_version("/opt/cuda-13.2"), Some(13));
        assert_eq!(parse_cuda_major_version("12.9"), Some(12));
        assert_eq!(parse_cuda_major_version("12_6"), Some(12));
        assert_eq!(parse_cuda_major_version("CUDA Version 13.2.1"), Some(13));
    }

    #[test]
    fn parse_cuda_major_from_cuda_version_metadata_supports_json_and_text() {
        assert_eq!(
            parse_cuda_major_from_cuda_version_metadata(
                r#"{"cuda":{"name":"CUDA SDK","version":"13.2.1"}}"#
            ),
            Some(13)
        );
        assert_eq!(
            parse_cuda_major_from_cuda_version_metadata("CUDA Version 12.8.0"),
            Some(12)
        );
    }

    #[test]
    fn detect_cuda_major_from_cuda_path_env_vars_ignores_unrelated_values() {
        let vars = [
            ("SHLVL", "1"),
            ("TERM_PROGRAM_VERSION", "3.5.1"),
            ("PATH", "/usr/bin"),
        ];
        assert_eq!(detect_cuda_major_from_cuda_path_env_vars(vars), None);
    }

    #[test]
    fn detect_cuda_major_from_cuda_path_env_vars_prefers_versioned_cuda_vars() {
        let vars = [
            ("SHLVL", "1"),
            ("CUDA_PATH_V13_2", "/opt/cuda"),
            ("TERM_PROGRAM_VERSION", "3.5.1"),
        ];
        assert_eq!(detect_cuda_major_from_cuda_path_env_vars(vars), Some(13));
    }

    #[test]
    fn detect_cuda_major_from_cuda_path_value_reads_cuda_version_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            temp_dir.path().join("version.json"),
            r#"{"cuda":{"version":"13.2.1"}}"#,
        )
        .unwrap();

        assert_eq!(
            detect_cuda_major_from_cuda_path_value(temp_dir.path().to_string_lossy().as_ref()),
            Some(13)
        );
    }

    #[test]
    fn parse_cuda_major_from_runtime_library_entry_handles_ldconfig_output() {
        assert_eq!(
            parse_cuda_major_from_runtime_library_entry(
                "libcudart.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcudart.so.13"
            ),
            Some(13)
        );
        assert_eq!(
            parse_cuda_major_from_runtime_library_entry("/opt/cuda/lib64/libcublasLt.so.13"),
            Some(13)
        );
        assert_eq!(
            parse_cuda_major_from_runtime_library_entry("libcufft.so.12"),
            None
        );
    }

    #[test]
    fn detect_cuda_major_from_library_entries_prefers_supported_runtime_libs() {
        let entries = [
            "libcufft.so.12 (libc6,x86-64) => /opt/cuda/lib64/libcufft.so.12",
            "libcublas.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcublas.so.13",
            "libcublasLt.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcublasLt.so.13",
            "libcudart.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcudart.so.13",
        ];
        assert_eq!(detect_cuda_major_from_library_entries(entries), Some(13));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn detect_cuda_major_from_library_dir_groups_respects_group_order() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cuda12_dir = temp_dir.path().join("cuda12");
        let cuda13_dir = temp_dir.path().join("cuda13");
        std::fs::create_dir_all(&cuda12_dir).unwrap();
        std::fs::create_dir_all(&cuda13_dir).unwrap();
        std::fs::write(cuda12_dir.join("libcudart.so.12"), b"").unwrap();
        std::fs::write(cuda13_dir.join("libcudart.so.13"), b"").unwrap();

        let groups = vec![vec![cuda12_dir], vec![cuda13_dir]];
        assert_eq!(detect_cuda_major_from_library_dir_groups(&groups), Some(12));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn detect_cuda_major_from_ldconfig_entries_filters_non_native_arch_entries() {
        let entries = [
            "libcudart.so.13 (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libcudart.so.13",
            "libcudart.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudart.so.12",
        ];
        assert_eq!(detect_cuda_major_from_ldconfig_entries(entries), Some(12));
    }

    #[test]
    fn cuda_ort_cache_dir_name_separates_cuda13_runtime() {
        assert_eq!(cuda_ort_cache_dir_name(None), "cuda");
        assert_eq!(cuda_ort_cache_dir_name(Some(12)), "cuda");
        assert_eq!(cuda_ort_cache_dir_name(Some(13)), "cuda13");
        assert_eq!(cuda_ort_cache_dir_name(Some(14)), "cuda13");
    }

    #[test]
    fn cached_ort_library_path_reuses_cuda13_cache_when_detection_is_unknown() {
        let temp_dir = tempfile::tempdir().unwrap();
        let expected_path = temp_dir
            .path()
            .join("lib")
            .join("cuda13")
            .join(platform_ort_lib_name());
        std::fs::create_dir_all(expected_path.parent().unwrap()).unwrap();
        std::fs::write(&expected_path, b"").unwrap();

        let resolved = cached_ort_library_path_for_ep_in_home(
            temp_dir.path(),
            OnnxExecutionProvider::Cuda,
            None,
        );
        assert_eq!(resolved, expected_path);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn ort_platform_info_uses_cuda13_archive_and_plain_internal_gpu_dir() {
        let (_, archive_name, internal_path, _, _) =
            ort_platform_info_with_cuda_major(OnnxExecutionProvider::Cuda, Some(13)).unwrap();
        assert!(archive_name.contains("-gpu_cuda13-"));
        assert!(internal_path.contains("onnxruntime-linux-x64-gpu-"));
        assert!(!internal_path.contains("_cuda13"));
    }

    #[test]
    fn wrap_ort_error_keeps_model_load_failures_specific() {
        let message =
            wrap_ort_error("failed to load embedding model C:\\Users\\me\\.vera\\model.onnx");
        assert!(message.contains("Failed to initialize ONNX session"));
        assert!(!message.contains("shared library not found"));
    }

    #[test]
    fn wrap_ort_error_still_flags_missing_dlls() {
        let message = wrap_ort_error(
            "LoadLibrary failed for onnxruntime.dll: The specified module could not be found",
        );
        assert!(message.contains("ONNX Runtime shared library not found"));
    }

    #[tokio::test]
    async fn test_download_failure_cleanup() {
        let temp_dir = tempfile::tempdir().unwrap();
        let home = temp_dir.path().join(".vera");

        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to bind test listener: {err}"),
        };
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Return a valid HTTP response header but truncate the body
                let response = "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nPartialData";
                let _ = stream.write_all(response.as_bytes());
                // abruptly close the connection
            }
        });

        let base_url = format!("http://127.0.0.1:{}", port);

        let res =
            ensure_model_file_impl("test-repo", "test-file.bin", &base_url, Some(&home)).await;

        assert!(res.is_err(), "Download should fail due to truncated stream");

        let target_dir = home.join("models").join("test-repo");
        let part_file = target_dir
            .join("test-file.bin")
            .with_extension(format!("part.{}", std::process::id()));
        assert!(
            !part_file.exists(),
            "Partial file should be cleaned up on failure"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_macos_rpath_loader_path_without_slash() {
        let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
        assert_eq!(
            resolve_macos_rpath("@loader_path", inspected),
            std::path::Path::new("/tmp/vera/lib")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_macos_rpath_loader_path_with_slash() {
        let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
        assert_eq!(
            resolve_macos_rpath("@loader_path/subdir", inspected),
            std::path::Path::new("/tmp/vera/lib/subdir")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_macos_rpath_executable_path_without_slash() {
        let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
        let expected = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
            .unwrap_or_else(|| std::path::PathBuf::from("@executable_path"));
        assert_eq!(resolve_macos_rpath("@executable_path", inspected), expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_macos_rpath_executable_path_with_slash() {
        let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
        let expected = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.join("subdir")))
            .unwrap_or_else(|| std::path::PathBuf::from("@executable_path/subdir"));
        assert_eq!(
            resolve_macos_rpath("@executable_path/subdir", inspected),
            expected
        );
    }
}
