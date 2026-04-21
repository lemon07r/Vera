//! `vera grep <pattern>` — Regex search over indexed files.

use anyhow::bail;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use crate::helpers::{load_runtime_config, output_results, warn_if_index_stale};

/// Run the `vera grep <pattern>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    pattern: &str,
    limit: Option<usize>,
    case_insensitive: bool,
    context_lines: usize,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
    timing: bool,
    git_scope: Option<vera_core::git_scope::GitScope>,
    compact: bool,
) -> anyhow::Result<()> {
    let config = load_runtime_config()?;
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    let mut filters = filters.clone();
    if let Some(scope) = git_scope.as_ref() {
        filters.exact_paths = Some(Arc::new(vera_core::git_scope::resolve_scope(&cwd, scope)?));
    }
    let index_dir = vera_core::indexing::index_dir(&cwd);

    if !index_dir.exists() {
        bail!(
            "no index found in current directory.\n\
             Hint: run `vera index <path>` first to create an index."
        );
    }
    warn_if_index_stale(&cwd, &config.indexing);

    let result_limit = limit.unwrap_or(20);
    let started_at = Instant::now();
    let results = vera_core::retrieval::search_regex(
        &index_dir,
        pattern,
        result_limit,
        case_insensitive,
        context_lines,
        &filters,
    )?;

    output_results(
        &results,
        json_output,
        raw,
        compact,
        config.retrieval.max_output_chars,
    );

    if results.is_empty() {
        if let Some(hint) = alternation_hint(pattern) {
            let stderr = std::io::stderr();
            let mut err = stderr.lock();
            let _ = writeln!(err, "{hint}");
        }
    }

    if timing {
        let elapsed = started_at.elapsed();
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        let _ = writeln!(err, "[timing] total: {}ms", elapsed.as_millis());
    }

    Ok(())
}

fn alternation_hint(pattern: &str) -> Option<String> {
    if !pattern.contains(r"\|") {
        return None;
    }

    let suggested = pattern.replace(r"\|", "|");
    if suggested == pattern {
        return None;
    }

    Some(format!(
        "hint: `vera grep` uses Rust regex syntax, so `\\|` matches a literal pipe. If you meant alternation, try `{suggested}`."
    ))
}

#[cfg(test)]
mod tests {
    use super::alternation_hint;

    #[test]
    fn alternation_hint_suggests_plain_pipe() {
        let hint =
            alternation_hint(r"persist_kimi_auth_record\|persist_factory_auth_record").unwrap();
        assert!(hint.contains("persist_kimi_auth_record|persist_factory_auth_record"));
    }

    #[test]
    fn alternation_hint_ignores_normal_patterns() {
        assert!(alternation_hint("TODO|FIXME").is_none());
    }
}
