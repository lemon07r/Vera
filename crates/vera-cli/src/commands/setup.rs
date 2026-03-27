//! `vera setup` — persist a preferred Vera mode and bootstrap first-run state.

use std::io::Write;

use anyhow::{Context, bail};
use serde::Serialize;
use vera_core::config::{InferenceBackend, OnnxExecutionProvider};

use crate::commands;
use crate::state::{self, ApiSetupInput};

#[derive(Debug, Serialize)]
pub(crate) struct SetupReport {
    mode: String,
    config_path: String,
    credentials_path: String,
    models_prefetched: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    onnx_runtime_ready: Option<bool>,
    indexed_path: Option<String>,
}

/// `backend`: Some(OnnxJina(..)) for local, None + api=true for API, None + api=false defaults to local CPU.
pub fn run(
    backend: Option<InferenceBackend>,
    api: bool,
    index_path: Option<String>,
    json_output: bool,
    yes: bool,
) -> anyhow::Result<()> {
    // Resolve: explicit backend flag wins, then --api, then interactive menu.
    let effective_backend = if api {
        InferenceBackend::Api
    } else if let Some(b) = backend {
        b
    } else if json_output || yes {
        // Non-interactive: auto-detect GPU, fall back to CPU.
        let detected = detect_gpu();
        if !json_output {
            eprintln!("Auto-detected backend: {detected}. Use a --onnx-jina-* flag to override.");
        }
        detected
    } else {
        // Interactive: show backend selection menu.
        prompt_backend()?
    };

    if !yes && !confirm(&effective_backend, index_path.as_deref())? {
        if !json_output {
            println!("Cancelled.");
        }
        return Ok(());
    }

    configure_backend(
        effective_backend,
        index_path,
        json_output,
        "Vera setup complete.",
    )
}

pub(crate) fn configure_backend(
    effective_backend: InferenceBackend,
    index_path: Option<String>,
    json_output: bool,
    success_header: &str,
) -> anyhow::Result<()> {
    let use_local = effective_backend.is_local();
    let mut models_prefetched = 0usize;
    let onnx_runtime_ready;

    if let InferenceBackend::OnnxJina(ep) = effective_backend {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
        let prefetched =
            rt.block_on(vera_core::local_models::prefetch_default_local_models_for_ep(ep))?;
        models_prefetched = prefetched.len();
        // Use the downloaded library path (first prefetched file) for the readiness check.
        onnx_runtime_ready = Some(
            vera_core::local_models::ensure_ort_runtime(prefetched.first().map(|p| p.as_path()))
                .is_ok(),
        );
        state::save_backend(effective_backend)?;
        state::apply_saved_env_force()?;
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
        commands::index::execute(path, effective_backend, Vec::new(), false, false)?;
    }

    let report = SetupReport {
        mode: effective_backend.to_string(),
        config_path: state::config_path()?.display().to_string(),
        credentials_path: state::credentials_path()?.display().to_string(),
        models_prefetched,
        onnx_runtime_ready,
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
                .args(["-c", "grep -rql 0x10de /sys/class/drm/*/device/vendor 2>/dev/null"])
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
            .args(["-c", "grep -rql 0x8086 /sys/class/drm/*/device/vendor 2>/dev/null"])
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

/// Show an interactive backend selection menu. Auto-detect is the default (press Enter).
fn prompt_backend() -> anyhow::Result<InferenceBackend> {
    let detected = detect_gpu();
    let detected_label = match detected {
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda) => "NVIDIA GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl) => "Apple Silicon detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm) => "AMD GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino) => "Intel GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl) => "DirectX 12 GPU assumed",
        _ => "no GPU detected, will use CPU",
    };

    println!("No backend specified. Select a backend:\n");
    println!("  [1] Auto-detect ({detected_label} -> {detected})  [default]");
    println!("  [2] API mode         (remote OpenAI-compatible endpoints)");
    println!("  [3] CUDA             (NVIDIA GPU)");
    println!("  [4] ROCm             (AMD GPU, Linux)");
    println!("  [5] CoreML           (Apple Silicon, macOS)");
    println!("  [6] OpenVINO         (Intel GPU/iGPU, Linux)");
    println!("  [7] DirectML         (DirectX 12 GPU, Windows)");
    println!("  [8] CPU              (slow, not recommended)");
    println!();
    print!("Choice [1]: ");
    std::io::stdout()
        .flush()
        .context("failed to flush prompt")?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read input")?;

    match input.trim() {
        "" | "1" => Ok(detected),
        "2" => Ok(InferenceBackend::Api),
        "3" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda)),
        "4" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm)),
        "5" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl)),
        "6" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino)),
        "7" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl)),
        "8" => Ok(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu)),
        other => bail!("invalid choice: {other}. Run `vera setup` again."),
    }
}

fn confirm(backend: &InferenceBackend, index_path: Option<&str>) -> anyhow::Result<bool> {
    println!("This will configure Vera for {backend} mode.");
    if let Some(path) = index_path {
        println!("It will also index: {path}");
    }
    print!("Continue? [y/N]: ");
    let mut stdout = std::io::stdout();
    std::io::Write::flush(&mut stdout).context("failed to flush confirmation prompt")?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read confirmation input")?;
    Ok(matches!(input.trim(), "y" | "Y" | "yes" | "YES"))
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
