//! Shared helper functions for CLI command implementations.

use clap::Args;
use serde::Serialize;

/// Load the effective runtime configuration.
pub fn load_runtime_config() -> anyhow::Result<vera_core::config::VeraConfig> {
    crate::state::load_runtime_config()
}

#[derive(Debug, Clone, Default, Args)]
pub struct LocalBackendFlags {
    /// Use local ONNX models on CPU.
    #[arg(long = "onnx-jina-cpu", group = "backend")]
    pub onnx_jina_cpu: bool,
    /// Use local ONNX models with CUDA (NVIDIA GPU).
    #[arg(long = "onnx-jina-cuda", group = "backend")]
    pub onnx_jina_cuda: bool,
    /// Use local ONNX models with ROCm (AMD GPU, Linux only).
    #[arg(long = "onnx-jina-rocm", group = "backend")]
    pub onnx_jina_rocm: bool,
    /// Use local ONNX models with DirectML (Windows GPU).
    #[arg(long = "onnx-jina-directml", group = "backend")]
    pub onnx_jina_directml: bool,
    /// Use local ONNX models with CoreML (Apple Silicon).
    #[arg(long = "onnx-jina-coreml", group = "backend")]
    pub onnx_jina_coreml: bool,
    /// Use local ONNX models with OpenVINO (Intel GPU/iGPU, Linux only).
    #[arg(long = "onnx-jina-openvino", group = "backend")]
    pub onnx_jina_openvino: bool,
    /// Alias for --onnx-jina-cpu (backwards compatibility).
    #[arg(long, group = "backend", hide = true)]
    pub local: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub struct GitScopeFlags {
    /// Limit results to modified, staged, and untracked files.
    #[arg(long, group = "git_scope")]
    pub changed: bool,
    /// Limit results to files changed since the given revision.
    #[arg(long, value_name = "REV", group = "git_scope")]
    pub since: Option<String>,
    /// Limit results to files changed since merge-base(HEAD, REV).
    #[arg(long, value_name = "REV", group = "git_scope")]
    pub base: Option<String>,
}

impl GitScopeFlags {
    pub fn resolve(&self) -> Option<vera_core::git_scope::GitScope> {
        if self.changed {
            Some(vera_core::git_scope::GitScope::Changed)
        } else if let Some(rev) = self.since.as_ref() {
            Some(vera_core::git_scope::GitScope::Since(rev.clone()))
        } else {
            self.base
                .as_ref()
                .map(|rev| vera_core::git_scope::GitScope::Base(rev.clone()))
        }
    }
}

impl LocalBackendFlags {
    pub fn any_set(&self) -> bool {
        self.onnx_jina_cpu
            || self.onnx_jina_cuda
            || self.onnx_jina_rocm
            || self.onnx_jina_directml
            || self.onnx_jina_coreml
            || self.onnx_jina_openvino
            || self.local
    }

    pub fn explicit_backend(&self) -> Option<vera_core::config::InferenceBackend> {
        self.any_set().then(|| resolve_backend_flags(self))
    }

    pub fn resolve(&self) -> vera_core::config::InferenceBackend {
        resolve_backend_flags(self)
    }
}

#[derive(Debug, Clone, Default, Args)]
pub struct LocalEmbeddingModelFlags {
    /// Use CodeRankEmbed instead of Vera's default Jina local embedding model.
    #[arg(
        long = "code-rank-embed",
        alias = "coderankembed",
        group = "local_embedding_source"
    )]
    pub code_rank_embed: bool,
    /// Hugging Face repo id or full Hugging Face URL for a custom local embedding model.
    #[arg(
        long = "embedding-repo",
        value_name = "REPO_OR_URL",
        group = "local_embedding_source"
    )]
    pub embedding_repo: Option<String>,
    /// Local directory containing a custom ONNX embedding model.
    #[arg(
        long = "embedding-dir",
        value_name = "DIR",
        group = "local_embedding_source"
    )]
    pub embedding_dir: Option<String>,
    /// Relative path to the ONNX file inside the selected repo or directory.
    #[arg(long = "embedding-onnx-file", value_name = "PATH")]
    pub embedding_onnx_file: Option<String>,
    /// Relative path to the ONNX external data file inside the selected repo or directory.
    #[arg(
        long = "embedding-onnx-data-file",
        value_name = "PATH",
        conflicts_with = "embedding_no_onnx_data"
    )]
    pub embedding_onnx_data_file: Option<String>,
    /// Use models that do not require an ONNX external data file.
    #[arg(long = "embedding-no-onnx-data")]
    pub embedding_no_onnx_data: bool,
    /// Relative path to the tokenizer file inside the selected repo or directory.
    #[arg(long = "embedding-tokenizer-file", value_name = "PATH")]
    pub embedding_tokenizer_file: Option<String>,
    /// Embedding dimension the model returns.
    #[arg(long = "embedding-dim", value_name = "DIM")]
    pub embedding_dim: Option<usize>,
    /// Pooling strategy for token-level output models.
    #[arg(long = "embedding-pooling", value_name = "POOLING", value_parser = ["mean", "cls"])]
    pub embedding_pooling: Option<String>,
    /// Tokenizer truncation length for local embedding inference.
    #[arg(long = "embedding-max-length", value_name = "TOKENS")]
    pub embedding_max_length: Option<usize>,
    /// Optional asymmetric query prefix for models that require it.
    #[arg(long = "embedding-query-prefix", value_name = "TEXT")]
    pub embedding_query_prefix: Option<String>,
}

impl LocalEmbeddingModelFlags {
    pub fn any_set(&self) -> bool {
        self.code_rank_embed
            || self.embedding_repo.is_some()
            || self.embedding_dir.is_some()
            || self.embedding_onnx_file.is_some()
            || self.embedding_onnx_data_file.is_some()
            || self.embedding_no_onnx_data
            || self.embedding_tokenizer_file.is_some()
            || self.embedding_dim.is_some()
            || self.embedding_pooling.is_some()
            || self.embedding_max_length.is_some()
            || self.embedding_query_prefix.is_some()
    }
}

/// Resolve an `InferenceBackend` from the per-command boolean flags.
pub fn resolve_backend_flags(flags: &LocalBackendFlags) -> vera_core::config::InferenceBackend {
    use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
    let explicit = if flags.onnx_jina_cpu || flags.local {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu))
    } else if flags.onnx_jina_cuda {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda))
    } else if flags.onnx_jina_rocm {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm))
    } else if flags.onnx_jina_directml {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl))
    } else if flags.onnx_jina_coreml {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl))
    } else if flags.onnx_jina_openvino {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino))
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

/// Truncate `content` to fit within `allowed` bytes, breaking at a line boundary.
fn truncate_to_budget(content: &str, allowed: usize) -> std::borrow::Cow<'_, str> {
    if content.len() <= allowed {
        return std::borrow::Cow::Borrowed(content);
    }
    let end = content
        .char_indices()
        .take_while(|(i, _)| *i < allowed)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let break_at = content[..end].rfind('\n').unwrap_or(end);
    let mut truncated = content[..break_at].to_string();
    truncated.push_str("\n[...truncated]");
    std::borrow::Cow::Owned(truncated)
}

/// Output search results with a total character budget.
///
/// Priority: `--json` compact JSON > `--raw` verbose > default markdown codeblocks.
/// When `budget` is non-zero, output is progressively truncated so the combined
/// content stays within the budget. Lower-ranked results are truncated first.
/// When `compact` is true, function/class bodies are stripped to show only signatures.
pub fn output_results(
    results: &[vera_core::types::SearchResult],
    json_output: bool,
    raw: bool,
    compact: bool,
    budget: usize,
) {
    use vera_core::parsing::signatures::extract_signature;

    // When compact mode is on, pre-compute signature-only content for each result.
    let compacted: Vec<String> = if compact {
        results
            .iter()
            .map(|r| extract_signature(&r.content, r.language))
            .collect()
    } else {
        Vec::new()
    };
    // Helper: pick compacted or original content by index.
    macro_rules! content_for {
        ($i:expr, $r:expr) => {
            if compact {
                compacted[$i].as_str()
            } else {
                $r.content.as_str()
            }
        };
    }

    if json_output {
        let json_results: Vec<CompactResult> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut cr = CompactResult::from_search_result(r);
                if compact {
                    cr.content = &compacted[i];
                }
                cr
            })
            .collect();
        let json = serde_json::to_string(&json_results)
            .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"));
        println!("{json}");
    } else if raw {
        if results.is_empty() {
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
                let display_content = content_for!(i, result);
                let preview: String = display_content
                    .lines()
                    .take(3)
                    .map(|l| format!("   │ {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                println!("{preview}");
                println!();
            }
        }
    } else {
        // Default: markdown codeblocks (most token-efficient for LLM agents).
        let mut remaining = budget;
        for (i, r) in results.iter().enumerate() {
            if budget > 0 && remaining == 0 {
                break;
            }
            if i > 0 {
                println!();
            }
            let mut info = format!("{}:{}-{}", r.file_path, r.line_start, r.line_end);
            if let (Some(stype), Some(name)) = (&r.symbol_type, &r.symbol_name) {
                info.push_str(&format!(" {stype}:{name}"));
            }
            println!("```{info}");
            let base_content = content_for!(i, r);
            let content = if budget > 0 {
                let c = truncate_to_budget(base_content, remaining);
                remaining = remaining.saturating_sub(c.len());
                c
            } else {
                std::borrow::Cow::Borrowed(base_content)
            };
            print!("{}", content);
            if !content.ends_with('\n') {
                println!();
            }
            println!("```");
        }
    }
}

/// Format a byte count as a compact human-readable string (e.g. "1.2 MB").
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1_024.0;
    const MB: f64 = 1_024.0 * KB;
    const GB: f64 = 1_024.0 * MB;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else {
        format!("{:.1} KB", b / KB)
    }
}

/// Print a human-readable summary of the indexing run.
///
/// When `verbose` is true, individual file paths are listed for skipped-file
/// categories. Otherwise only counts are shown with a hint to rerun with `-v`.
pub fn print_human_summary(summary: &vera_core::indexing::IndexSummary, verbose: bool) {
    println!("Indexing complete!");
    println!();
    println!("  Files parsed:        {}", summary.files_parsed);
    println!("  Chunks created:      {}", summary.chunks_created);
    println!("  Embeddings generated: {}", summary.embeddings_generated);
    println!("  Elapsed time:        {:.2}s", summary.elapsed_secs);

    if summary.files_with_tree_sitter_errors > 0 || summary.files_using_tier0_fallback > 0 {
        println!();
        println!("  Index health:");
        if summary.files_with_tree_sitter_errors > 0 {
            println!(
                "    Tree-sitter errors: {}",
                summary.files_with_tree_sitter_errors
            );
        }
        if summary.files_using_tier0_fallback > 0 {
            println!(
                "    Tier 0 fallback:    {}",
                summary.files_using_tier0_fallback
            );
        }
    }

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
            if verbose {
                for (path, size) in &summary.large_skipped_paths {
                    println!("      - {path} ({size})", size = format_bytes(*size));
                }
            }
        }
        if summary.error_skipped > 0 {
            println!("    Read errors: {}", summary.error_skipped);
        }
        if !verbose && !summary.large_skipped_paths.is_empty() {
            println!();
            println!("  Rerun with --verbose (-v) to see skipped file paths.");
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
