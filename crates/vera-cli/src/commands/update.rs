//! `vera update <path>` — Incrementally update the index.

use std::path::Path;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;

use crate::helpers::load_runtime_config;

/// Run the `vera update <path>` command.
pub fn run(
    path: &str,
    json_output: bool,
    backend: InferenceBackend,
    exclude: Vec<String>,
    no_ignore: bool,
    no_default_excludes: bool,
) -> anyhow::Result<()> {
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
             Hint: vera update expects a directory path, not a file."
        );
    }

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);
    config.indexing.extra_excludes = exclude;
    config.indexing.no_ignore = no_ignore;
    config.indexing.no_default_excludes = no_default_excludes;

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    // Check metadata mismatch
    let metadata_path = repo_path.join(".vera").join("metadata.db");
    if let Ok(metadata_store) = vera_core::storage::metadata::MetadataStore::open(&metadata_path) {
        if let (Some(s_model), Some(s_dim)) = (
            metadata_store.get_index_meta("model_name").unwrap_or(None),
            metadata_store
                .get_index_meta("embedding_dim")
                .unwrap_or(None),
        ) {
            if !vera_core::config::model_names_match(&s_model, &model_name) {
                bail!(
                    "Index was created with model '{}' ({} dimensions), but you are using model '{}'. Please re-index with matching provider.",
                    s_model,
                    s_dim,
                    model_name
                );
            }
            if let Ok(dim) = s_dim.parse::<usize>() {
                use vera_core::embedding::EmbeddingProvider;
                if let Some(provider_dim) = provider.expected_dim() {
                    if provider_dim < dim {
                        bail!(
                            "Dimension mismatch: index has {} dimensions but active provider only returns {}. Please re-index with matching provider.",
                            dim,
                            provider_dim
                        );
                    }
                }
            }
        }
    }

    // Run the incremental update pipeline.
    let summary = rt
        .block_on(vera_core::indexing::update_repository(
            repo_path,
            &provider,
            &config,
            &model_name,
        ))
        .context("update failed")?;

    // Output results.
    if json_output {
        let json = serde_json::to_string_pretty(&summary)
            .map_err(|e| anyhow::anyhow!("failed to serialize summary: {e}"))?;
        println!("{json}");
    } else {
        print_update_summary(&summary);
    }

    Ok(())
}

/// Print a human-readable summary of the update run.
fn print_update_summary(summary: &vera_core::indexing::UpdateSummary) {
    println!("Update complete!");
    println!();
    println!("  Files modified:  {}", summary.files_modified);
    println!("  Files added:     {}", summary.files_added);
    println!("  Files deleted:   {}", summary.files_deleted);
    println!("  Files unchanged: {}", summary.files_unchanged);
    if summary.files_with_tree_sitter_errors > 0 || summary.files_using_tier0_fallback > 0 {
        println!(
            "  Tree-sitter errors: {}",
            summary.files_with_tree_sitter_errors
        );
        println!(
            "  Tier 0 fallback:    {}",
            summary.files_using_tier0_fallback
        );
    }
    println!("  Total chunks:    {}", summary.total_chunks);
    println!("  Elapsed time:    {:.2}s", summary.elapsed_secs);

    if !summary.parse_errors.is_empty() {
        println!();
        println!("  Parse errors ({}):", summary.parse_errors.len());
        for err in &summary.parse_errors {
            println!("    {}: {}", err.file_path, err.error);
        }
    }

    let total_changed = summary.files_modified + summary.files_added + summary.files_deleted;
    if total_changed == 0 {
        println!();
        println!("  Index is up to date — no changes detected.");
    }
}
