//! `vera setup` — persist a preferred Vera mode and bootstrap first-run state.

use anyhow::{Context, bail};
use serde::Serialize;
use vera_core::config::{InferenceBackend, OnnxExecutionProvider};

use crate::commands;
use crate::state::{self, ApiSetupInput};

#[derive(Debug, Serialize)]
struct SetupReport {
    mode: String,
    config_path: String,
    credentials_path: String,
    models_prefetched: usize,
    onnx_runtime_ready: bool,
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
    // Resolve: explicit backend flag wins, then --api, then default to local CPU.
    let effective_backend = if api {
        InferenceBackend::Api
    } else {
        backend.unwrap_or(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu))
    };

    let use_local = effective_backend.is_local();

    if !yes && !confirm(&effective_backend, index_path.as_deref())? {
        if !json_output {
            println!("Cancelled.");
        }
        return Ok(());
    }

    let mut models_prefetched = 0usize;
    let onnx_runtime_ready;

    if let InferenceBackend::OnnxJina(ep) = effective_backend {
        state::save_local_mode(true)?;
        state::apply_saved_env_force()?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
        let prefetched = rt.block_on(vera_core::local_models::prefetch_default_local_models_for_ep(ep))?;
        models_prefetched = prefetched.len();
        // Use the downloaded library path (first prefetched file) for the readiness check.
        onnx_runtime_ready = vera_core::local_models::ensure_ort_runtime(
            prefetched.first().map(|p| p.as_path()),
        )
        .is_ok();
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
        state::save_local_mode(false)?;
        state::apply_saved_env_force()?;
        onnx_runtime_ready = vera_core::local_models::ensure_ort_runtime(None).is_ok();
    }

    if let Some(path) = index_path.as_deref() {
        commands::index::execute(path, effective_backend)?;
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
        println!("Vera setup complete.");
        println!();
        println!("  Mode:                 {}", report.mode);
        println!("  Config:               {}", report.config_path);
        println!("  Credentials:          {}", report.credentials_path);
        if use_local {
            println!("  Prefetched model files: {}", report.models_prefetched);
        }
        println!(
            "  ONNX Runtime ready:   {}",
            if report.onnx_runtime_ready {
                "yes"
            } else {
                "no"
            }
        );
        if let Some(path) = report.indexed_path.as_deref() {
            println!("  Indexed path:         {path}");
        }
    }

    Ok(())
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
