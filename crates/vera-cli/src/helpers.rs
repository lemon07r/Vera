//! Shared helper functions for CLI command implementations.

use serde::Serialize;

/// Load the effective runtime configuration.
pub fn load_runtime_config() -> anyhow::Result<vera_core::config::VeraConfig> {
    crate::state::load_runtime_config()
}

/// Resolve an `InferenceBackend` from the per-command boolean flags.
pub fn resolve_backend_flags(
    onnx_jina_cpu: bool,
    onnx_jina_cuda: bool,
    onnx_jina_rocm: bool,
    onnx_jina_directml: bool,
    local: bool,
) -> vera_core::config::InferenceBackend {
    use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
    let explicit = if onnx_jina_cpu || local {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu))
    } else if onnx_jina_cuda {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda))
    } else if onnx_jina_rocm {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm))
    } else if onnx_jina_directml {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl))
    } else {
        None
    };
    vera_core::config::resolve_backend(explicit)
}

/// Compact JSON representation that drops low-signal fields (`score`, `language`)
/// and omits null optional fields. This is the default for AI agent consumption.
#[derive(Serialize)]
struct CompactResult<'a> {
    file_path: &'a str,
    line_start: u32,
    line_end: u32,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_type: Option<&'a vera_core::types::SymbolType>,
}

impl<'a> CompactResult<'a> {
    fn from_search_result(r: &'a vera_core::types::SearchResult) -> Self {
        Self {
            file_path: &r.file_path,
            line_start: r.line_start,
            line_end: r.line_end,
            content: &r.content,
            symbol_name: r.symbol_name.as_deref(),
            symbol_type: r.symbol_type.as_ref(),
        }
    }
}

/// Output search results in human-readable or JSON format.
///
/// When `raw` is true, outputs the full `SearchResult` with all fields
/// (pretty-printed JSON or verbose human text). When false (default),
/// outputs compact single-line JSON optimized for AI agent token budgets.
pub fn output_results(results: &[vera_core::types::SearchResult], json_output: bool, raw: bool) {
    if json_output || !raw {
        if raw {
            // --raw --json: full pretty-printed output with all fields
            let json = serde_json::to_string_pretty(results)
                .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"));
            println!("{json}");
        } else {
            // Default: compact single-line JSON, no score/language/nulls
            let compact: Vec<CompactResult> =
                results.iter().map(CompactResult::from_search_result).collect();
            let json = serde_json::to_string(&compact)
                .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"));
            println!("{json}");
        }
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
