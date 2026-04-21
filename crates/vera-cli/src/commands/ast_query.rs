//! `vera ast-query <query>` — Structural search with raw tree-sitter queries.

use anyhow::{Context, bail};
use std::io::Write;
use std::time::Instant;

use crate::helpers::{load_runtime_config, output_results, warn_if_index_stale};

#[allow(clippy::too_many_arguments)]
pub fn run(
    query: &str,
    language: &str,
    path: Option<String>,
    limit: Option<usize>,
    scope: Option<String>,
    include_generated: bool,
    json_output: bool,
    raw: bool,
    timing: bool,
    compact: bool,
) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    let index_dir = vera_core::indexing::index_dir(&cwd);
    if !index_dir.exists() {
        bail!(
            "no index found in current directory.\n\
             Hint: run `vera index <path>` first to create an index."
        );
    }

    let language: vera_core::types::Language = language.parse().map_err(|_| {
        anyhow::anyhow!(
            "unsupported language '{}'. Use Vera's lowercase language names such as rust, python, or typescript.",
            language
        )
    })?;

    let filters = vera_core::types::SearchFilters {
        language: None,
        path_glob: path,
        exact_paths: None,
        symbol_type: None,
        scope: scope.and_then(|value| value.parse().ok()),
        include_generated: Some(include_generated),
    };

    let config = load_runtime_config()?;
    let result_limit = limit.unwrap_or(20);
    let started_at = Instant::now();
    warn_if_index_stale(&cwd, &config.indexing);
    let results =
        vera_core::retrieval::search_ast_query(&index_dir, query, language, result_limit, &filters)
            .context("AST query failed")?;

    output_results(
        &results,
        json_output,
        raw,
        compact,
        config.retrieval.max_output_chars,
    );

    if timing {
        let elapsed = started_at.elapsed();
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        let _ = writeln!(err, "[timing] total: {}ms", elapsed.as_millis());
    }

    Ok(())
}
