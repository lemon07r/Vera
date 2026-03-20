//! `vera index <path>` — Index a codebase for search.

use std::path::Path;
use std::process;

use crate::helpers::{create_embedding_provider, print_human_summary};

/// Run the `vera index <path>` command.
pub fn run(path: &str, json_output: bool) -> anyhow::Result<()> {
    let repo_path = Path::new(path);

    // Validate path early — before requiring API credentials.
    if !repo_path.exists() {
        eprintln!(
            "Error: path does not exist: {path}\n\
             Hint: check the path and try again."
        );
        process::exit(1);
    }
    if !repo_path.is_dir() {
        eprintln!(
            "Error: path is not a directory: {path}\n\
             Hint: vera index expects a directory path, not a file."
        );
        process::exit(1);
    }

    // Build the tokio runtime for async embedding calls.
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let config = vera_core::config::VeraConfig::default();

    // Create the embedding provider from environment.
    let provider = create_embedding_provider(&config)?;

    // Run the indexing pipeline.
    let summary = match rt.block_on(vera_core::indexing::index_repository(
        repo_path, &provider, &config,
    )) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Error: indexing failed: {err:#}");
            process::exit(1);
        }
    };

    // Output results.
    if json_output {
        let json = serde_json::to_string_pretty(&summary)
            .map_err(|e| anyhow::anyhow!("failed to serialize summary: {e}"))?;
        println!("{json}");
    } else {
        print_human_summary(&summary);
    }

    Ok(())
}
