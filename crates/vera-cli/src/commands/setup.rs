//! `vera setup` — persist a preferred Vera mode and bootstrap first-run state.

use anyhow::{Context, bail};
use serde::Serialize;
use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
use vera_core::local_models::{
    LocalEmbeddingModelConfig, LocalEmbeddingPooling, normalize_huggingface_repo,
};

use crate::commands;
use crate::helpers::LocalEmbeddingModelFlags;
use crate::state::{self, ApiSetupInput};

#[derive(Debug, Serialize)]
pub(crate) struct SetupReport {
    mode: String,
    config_path: String,
    credentials_path: String,
    models_prefetched: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    onnx_runtime_ready: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_embedding_model: Option<String>,
    indexed_path: Option<String>,
}

/// `backend`: Some(OnnxJina(..)) for local, None + api=true for API, None + api=false defaults to local CPU.
pub fn run(
    backend: Option<InferenceBackend>,
    api: bool,
    index_path: Option<String>,
    json_output: bool,
    yes: bool,
    embedding_flags: LocalEmbeddingModelFlags,
) -> anyhow::Result<()> {
    // If no flags at all and interactive, run the full wizard.
    let is_bare_interactive =
        !api && backend.is_none() && !json_output && !yes && !embedding_flags.any_set();
    if is_bare_interactive && index_path.is_none() {
        return run_wizard();
    }

    // Resolve: explicit backend flag wins, then --api, then auto-detect.
    let effective_backend = if api {
        InferenceBackend::Api
    } else if let Some(b) = backend {
        b
    } else if json_output || yes {
        let detected = detect_gpu();
        if !json_output {
            eprintln!("Auto-detected backend: {detected}. Use a --onnx-jina-* flag to override.");
        }
        detected
    } else {
        prompt_backend()?
    };
    if !effective_backend.is_local() && embedding_flags.any_set() {
        bail!("custom local embedding flags can only be used with local ONNX backends");
    }
    let local_embedding_model = effective_backend
        .is_local()
        .then(|| resolve_local_embedding_model(&embedding_flags))
        .transpose()?;

    if !yes
        && !confirm(
            &effective_backend,
            local_embedding_model.as_ref(),
            index_path.as_deref(),
        )?
    {
        if !json_output {
            println!("Cancelled.");
        }
        return Ok(());
    }

    configure_backend(
        effective_backend,
        local_embedding_model,
        index_path,
        false,
        "Vera setup complete.",
    )
}

/// Full interactive setup wizard: backend, agent skills, optional indexing.
fn run_wizard() -> anyhow::Result<()> {
    cliclack::intro("vera setup")?;

    // Step 1: Backend selection
    cliclack::log::step("Step 1: Backend")?;
    let effective_backend = prompt_backend_select()?;
    let local_embedding_model = effective_backend
        .is_local()
        .then(LocalEmbeddingModelConfig::default);

    configure_backend(
        effective_backend,
        local_embedding_model,
        None,
        false,
        "Backend configured.",
    )?;

    // Step 2: Agent skill installation
    cliclack::log::step("Step 2: Agent skills")?;
    let install_skills: bool = cliclack::confirm("Install Vera skills for coding agents?")
        .initial_value(true)
        .interact()?;
    if install_skills {
        commands::agent::run(commands::agent::AgentCommand::Install, None, None, false)?;
    }

    // Step 3: Optional indexing
    cliclack::log::step("Step 3: Index a project")?;
    let index_now: bool = cliclack::confirm("Index a project now?")
        .initial_value(false)
        .interact()?;
    if index_now {
        let path: String = cliclack::input("Project path")
            .default_input(".")
            .interact()?;
        commands::index::execute(
            path.trim(),
            effective_backend,
            Vec::new(),
            false,
            false,
            false,
        )?;
    }

    cliclack::outro("Setup complete! Run `vera search \"query\"` to get started.")?;
    Ok(())
}

pub(crate) fn configure_backend(
    effective_backend: InferenceBackend,
    local_embedding_model: Option<LocalEmbeddingModelConfig>,
    index_path: Option<String>,
    json_output: bool,
    success_header: &str,
) -> anyhow::Result<()> {
    let use_local = effective_backend.is_local();
    let mut models_prefetched = 0usize;
    let onnx_runtime_ready;
    let mut local_embedding_summary = None;

    if let InferenceBackend::OnnxJina(ep) = effective_backend {
        let local_embedding_model = local_embedding_model.unwrap_or_default();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
        let prefetched = rt.block_on(vera_core::local_models::prepare_local_models_for_ep(
            ep,
            &local_embedding_model,
        ))?;
        models_prefetched = prefetched.len();
        // Use the downloaded library path (first prefetched file) for the readiness check.
        onnx_runtime_ready = Some(
            vera_core::local_models::ensure_ort_runtime(prefetched.first().map(|p| p.as_path()))
                .is_ok(),
        );
        state::save_backend(effective_backend)?;
        state::save_local_embedding_model(&local_embedding_model)?;
        state::apply_saved_env_force()?;
        local_embedding_summary = Some(local_embedding_model.display_name());
    } else {
        let embedding = read_required_api_env(
            "EMBEDDING_MODEL_BASE_URL",
            "EMBEDDING_MODEL_ID",
            "EMBEDDING_MODEL_API_KEY",
        )?;
        let reranker = read_optional_api_env(
            "RERANKER_MODEL_BASE_URL",
            "RERANKER_MODEL_ID",
            "RERANKER_MODEL_API_KEY",
        )?;
        state::save_api_setup(&embedding, reranker.as_ref())?;
        state::apply_saved_env_force()?;
        onnx_runtime_ready = None;
    }

    if state::load_saved_config()?.install_method.is_none() {
        if let Some(install_method) = crate::update_check::resolve_install_method().install_method {
            state::save_install_method(Some(&install_method))?;
        }
    }

    if let Some(path) = index_path.as_deref() {
        commands::index::execute(path, effective_backend, Vec::new(), false, false, false)?;
    }

    let report = SetupReport {
        mode: effective_backend.to_string(),
        config_path: state::config_path()?.display().to_string(),
        credentials_path: state::credentials_path()?.display().to_string(),
        models_prefetched,
        onnx_runtime_ready,
        local_embedding_model: local_embedding_summary,
        indexed_path: index_path,
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{success_header}");
        println!();
        println!("  Mode:                 {}", report.mode);
        println!("  Config:               {}", report.config_path);
        println!("  Credentials:          {}", report.credentials_path);
        if use_local {
            if let Some(model) = report.local_embedding_model.as_deref() {
                println!("  Embedding model:      {model}");
            }
            println!("  Prefetched model files: {}", report.models_prefetched);
            println!(
                "  ONNX Runtime ready:   {}",
                if report.onnx_runtime_ready == Some(true) {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        if let Some(path) = report.indexed_path.as_deref() {
            println!("  Indexed path:         {path}");
        }
    }

    Ok(())
}

/// Probe the system for a usable GPU and return the best local backend.
/// Falls back to CPU if nothing is detected.
fn detect_gpu() -> InferenceBackend {
    // NVIDIA: check for nvidia-smi or vendor ID (0x10de) in sysfs
    let has_nvidia = std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
        || (cfg!(target_os = "linux")
            && std::process::Command::new("sh")
                .args([
                    "-c",
                    "grep -rql 0x10de /sys/class/drm/*/device/vendor 2>/dev/null",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success()));
    if has_nvidia {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda);
    }

    // Apple Silicon: macOS + aarch64
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl);
    }

    // AMD ROCm: check for rocminfo (Linux only)
    if cfg!(target_os = "linux")
        && std::process::Command::new("rocminfo")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm);
    }

    // Intel OpenVINO: check for Intel GPU via vendor ID (0x8086) in sysfs
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        let has_intel_gpu = std::process::Command::new("sh")
            .args([
                "-c",
                "grep -rql 0x8086 /sys/class/drm/*/device/vendor 2>/dev/null",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if has_intel_gpu {
            return InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino);
        }
    }

    // Windows: DirectML works with any DirectX 12 GPU
    if cfg!(target_os = "windows") {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl);
    }

    InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu)
}

/// Show an interactive backend selection menu. Auto-detect is the default.
fn prompt_backend() -> anyhow::Result<InferenceBackend> {
    cliclack::intro("vera backend")?;
    let backend = prompt_backend_select()?;
    Ok(backend)
}

/// Backend selection menu items (no intro/outro, for embedding in wizards).
fn prompt_backend_select() -> anyhow::Result<InferenceBackend> {
    let detected = detect_gpu();
    let detected_hint = match detected {
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda) => "NVIDIA GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl) => "Apple Silicon detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm) => "AMD GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino) => "Intel GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl) => "DirectX 12 GPU assumed",
        _ => "no GPU detected, will use CPU",
    };

    let backend: InferenceBackend = cliclack::select("Select a backend")
        .item(
            detected,
            format!("Auto-detect ({detected_hint})"),
            "recommended",
        )
        .item(
            InferenceBackend::Api,
            "API mode",
            "remote OpenAI-compatible endpoints",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda),
            "CUDA",
            "NVIDIA GPU",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm),
            "ROCm",
            "AMD GPU, Linux",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl),
            "CoreML",
            "Apple Silicon, macOS",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino),
            "OpenVINO",
            "Intel GPU/iGPU, Linux",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl),
            "DirectML",
            "DirectX 12 GPU, Windows",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu),
            "CPU",
            "slow, not recommended",
        )
        .interact()?;

    Ok(backend)
}

fn resolve_local_embedding_model(
    flags: &LocalEmbeddingModelFlags,
) -> anyhow::Result<LocalEmbeddingModelConfig> {
    let mut model = if flags.code_rank_embed {
        LocalEmbeddingModelConfig::coderankembed()
    } else if let Some(repo_or_url) = flags.embedding_repo.as_deref() {
        LocalEmbeddingModelConfig::from_huggingface_repo(normalize_huggingface_repo(repo_or_url)?)
    } else if let Some(dir) = flags.embedding_dir.as_deref() {
        let path = std::path::Path::new(dir)
            .canonicalize()
            .with_context(|| format!("failed to resolve embedding directory: {dir}"))?;
        LocalEmbeddingModelConfig::from_directory(path)
    } else {
        LocalEmbeddingModelConfig::default()
    };

    if let Some(onnx_file) = flags.embedding_onnx_file.as_ref() {
        model.onnx_file = onnx_file.clone();
    }
    if flags.embedding_no_onnx_data {
        model.onnx_data_file = None;
    } else if let Some(onnx_data_file) = flags.embedding_onnx_data_file.as_ref() {
        model.onnx_data_file = Some(onnx_data_file.clone());
    }
    if let Some(tokenizer_file) = flags.embedding_tokenizer_file.as_ref() {
        model.tokenizer_file = tokenizer_file.clone();
    }
    if let Some(dim) = flags.embedding_dim {
        model.embedding_dim = dim;
    }
    if let Some(pooling) = flags.embedding_pooling.as_deref() {
        model.pooling = pooling
            .parse::<LocalEmbeddingPooling>()
            .map_err(anyhow::Error::msg)?;
    }
    if let Some(max_length) = flags.embedding_max_length {
        model.max_length = max_length;
    }
    if let Some(query_prefix) = flags.embedding_query_prefix.as_ref() {
        model.query_prefix =
            Some(query_prefix.trim().to_string()).filter(|value| !value.is_empty());
    }

    Ok(model)
}

fn confirm(
    backend: &InferenceBackend,
    local_embedding_model: Option<&LocalEmbeddingModelConfig>,
    index_path: Option<&str>,
) -> anyhow::Result<bool> {
    let mut msg = format!("Configure Vera for {backend} mode");
    if let Some(model) = local_embedding_model {
        msg.push_str(&format!(", embedding model: {}", model.display_name()));
    }
    if let Some(path) = index_path {
        msg.push_str(&format!(", then index: {path}"));
    }
    msg.push('?');
    let yes: bool = cliclack::confirm(msg).interact()?;
    Ok(yes)
}

fn read_required_api_env(
    base_key: &str,
    model_key: &str,
    api_key_key: &str,
) -> anyhow::Result<ApiSetupInput> {
    Ok(ApiSetupInput {
        base_url: std::env::var(base_key)
            .with_context(|| format!("{base_key} must be set for `vera setup --api`"))?,
        model_id: std::env::var(model_key)
            .with_context(|| format!("{model_key} must be set for `vera setup --api`"))?,
        api_key: std::env::var(api_key_key)
            .with_context(|| format!("{api_key_key} must be set for `vera setup --api`"))?,
    })
}

fn read_optional_api_env(
    base_key: &str,
    model_key: &str,
    api_key_key: &str,
) -> anyhow::Result<Option<ApiSetupInput>> {
    let base = std::env::var(base_key).ok();
    let model = std::env::var(model_key).ok();
    let api_key = std::env::var(api_key_key).ok();

    match (base, model, api_key) {
        (Some(base_url), Some(model_id), Some(api_key)) => Ok(Some(ApiSetupInput {
            base_url,
            model_id,
            api_key,
        })),
        (None, None, None) => Ok(None),
        _ => bail!(
            "reranker config is incomplete. Set all of {base_key}, {model_key}, and {api_key_key}, or leave all three unset."
        ),
    }
}
