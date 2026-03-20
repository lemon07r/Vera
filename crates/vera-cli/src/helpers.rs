//! Shared helper functions for CLI command implementations.

use std::process;

/// Create an embedding provider from environment variables.
pub fn create_embedding_provider(
    config: &vera_core::config::VeraConfig,
) -> anyhow::Result<vera_core::embedding::OpenAiProvider> {
    let provider_config = match vera_core::embedding::EmbeddingProviderConfig::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!(
                "Error: embedding API not configured: {err}\n\
                 Hint: set EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and \
                 EMBEDDING_MODEL_API_KEY environment variables."
            );
            process::exit(1);
        }
    };
    let provider_config = provider_config
        .with_timeout(std::time::Duration::from_secs(
            config.embedding.timeout_secs,
        ))
        .with_max_retries(config.embedding.max_retries);

    match vera_core::embedding::OpenAiProvider::new(provider_config) {
        Ok(p) => Ok(p),
        Err(err) => {
            eprintln!("Error: failed to initialize embedding provider: {err}");
            process::exit(1);
        }
    }
}

/// Output search results in human-readable or JSON format.
pub fn output_results(results: &[vera_core::types::SearchResult], json_output: bool) {
    if json_output {
        let json = serde_json::to_string_pretty(results)
            .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"));
        println!("{json}");
    } else if results.is_empty() {
        println!("No results found.");
    } else {
        for (i, result) in results.iter().enumerate() {
            println!(
                "{}. {} (lines {}-{}, {})",
                i + 1,
                result.file_path,
                result.line_start,
                result.line_end,
                result.language,
            );
            if let Some(ref name) = result.symbol_name {
                if let Some(ref stype) = result.symbol_type {
                    println!("   {stype} {name}");
                } else {
                    println!("   {name}");
                }
            }
            println!("   score: {:.6}", result.score);

            // Show a preview of the content (first 3 lines).
            let preview: String = result
                .content
                .lines()
                .take(3)
                .map(|l| format!("   │ {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            println!("{preview}");
            println!();
        }
    }
}

/// Print a human-readable summary of the indexing run.
pub fn print_human_summary(summary: &vera_core::indexing::IndexSummary) {
    println!("Indexing complete!");
    println!();
    println!("  Files parsed:        {}", summary.files_parsed);
    println!("  Chunks created:      {}", summary.chunks_created);
    println!("  Embeddings generated: {}", summary.embeddings_generated);
    println!("  Elapsed time:        {:.2}s", summary.elapsed_secs);

    // Report skipped files if any.
    let skipped_total = summary.binary_skipped + summary.large_skipped + summary.error_skipped;
    if skipped_total > 0 {
        println!();
        println!("  Skipped files:");
        if summary.binary_skipped > 0 {
            println!("    Binary:     {}", summary.binary_skipped);
        }
        if summary.large_skipped > 0 {
            println!("    Too large:  {}", summary.large_skipped);
        }
        if summary.error_skipped > 0 {
            println!("    Read errors: {}", summary.error_skipped);
        }
    }

    // Report parse errors if any.
    if !summary.parse_errors.is_empty() {
        println!();
        println!("  Parse errors ({}):", summary.parse_errors.len());
        for err in &summary.parse_errors {
            println!("    {}: {}", err.file_path, err.error);
        }
    }

    // Special message for empty repos.
    if summary.files_parsed == 0 && summary.chunks_created == 0 {
        println!();
        println!("  No source files found to index.");
    }
}
