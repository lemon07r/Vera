//! `vera grep <pattern>` — Regex search over indexed files.

use anyhow::bail;

use crate::helpers::output_results;

/// Run the `vera grep <pattern>` command.
pub fn run(
    pattern: &str,
    limit: Option<usize>,
    case_insensitive: bool,
    context_lines: usize,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
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

    let result_limit = limit.unwrap_or(20);
    let results = vera_core::retrieval::search_regex(
        &index_dir,
        pattern,
        result_limit,
        case_insensitive,
        context_lines,
        filters,
    )?;

    output_results(&results, json_output, raw);
    Ok(())
}
