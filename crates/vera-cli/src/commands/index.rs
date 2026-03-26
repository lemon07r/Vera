//! `vera index <path>` — Index a codebase for search.

use std::path::Path;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;

use crate::helpers::{load_runtime_config, print_human_summary};

/// Run the `vera index <path>` command.
pub fn run(path: &str, json_output: bool, backend: InferenceBackend) -> anyhow::Result<()> {
    let summary = execute(path, backend)?;

    if json_output {
        let json = serde_json::to_string_pretty(&summary)
            .map_err(|e| anyhow::anyhow!("failed to serialize summary: {e}"))?;
        println!("{json}");
    } else {
        print_human_summary(&summary);
    }

    Ok(())
}

/// Index a repository and return the resulting summary.
pub fn execute(path: &str, backend: InferenceBackend) -> anyhow::Result<vera_core::indexing::IndexSummary> {
    let repo_path = Path::new(path);

    if !repo_path.exists() {
        bail!(
            "path does not exist: {path}\n\
             Hint: check the path and try again."
        );
    }
    if !repo_path.is_dir() {
        bail!(
            "path is not a directory: {path}\n\
             Hint: vera index expects a directory path, not a file."
        );
    }

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    let summary = rt
        .block_on(vera_core::indexing::index_repository(
            repo_path,
            &provider,
            &config,
            &model_name,
        ))
        .context("indexing failed")?;

    Ok(summary)
}
