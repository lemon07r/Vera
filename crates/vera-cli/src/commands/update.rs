//! `vera update <path>` — Incrementally update the index.

use std::path::Path;
use std::process;

use crate::helpers::create_embedding_provider;

/// Run the `vera update <path>` command.
pub fn run(path: &str, json_output: bool) -> anyhow::Result<()> {
    let repo_path = Path::new(path);

    // Validate path early.
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
             Hint: vera update expects a directory path, not a file."
        );
        process::exit(1);
    }

    // Build the tokio runtime for async embedding calls.
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let config = vera_core::config::VeraConfig::default();

    // Create the embedding provider from environment.
    let provider = create_embedding_provider(&config)?;

    // Run the incremental update pipeline.
    let summary = match rt.block_on(vera_core::indexing::update_repository(
        repo_path, &provider, &config,
    )) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Error: update failed: {err:#}");
            process::exit(1);
        }
    };

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
    println!("  Total chunks:    {}", summary.total_chunks);
    println!("  Elapsed time:    {:.2}s", summary.elapsed_secs);

    let total_changed = summary.files_modified + summary.files_added + summary.files_deleted;
    if total_changed == 0 {
        println!();
        println!("  Index is up to date — no changes detected.");
    }
}
