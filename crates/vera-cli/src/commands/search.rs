//! `vera search <query>` — Search the indexed codebase.

use anyhow::bail;
use vera_core::config::InferenceBackend;

use crate::helpers::{load_runtime_config, output_results};

/// Run the `vera search <query>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    query: &str,
    limit: Option<usize>,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
    timing: bool,
    backend: InferenceBackend,
) -> anyhow::Result<()> {
    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);

    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    let index_dir = vera_core::indexing::index_dir(&cwd);

    if !index_dir.exists() {
        bail!(
            "no index found in current directory.\n\
             Hint: run `vera index <path>` first to create an index."
        );
    }

    let (results, timings) = vera_core::retrieval::search_service::execute_search(
        &index_dir,
        query,
        &config,
        filters,
        result_limit,
        backend,
    )?;

    output_results(&results, json_output, raw);

    if timing {
        use std::io::Write;
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        let fmt = |d: Option<std::time::Duration>| -> String {
            match d {
                Some(d) => format!("{}ms", d.as_millis()),
                None => "n/a".to_string(),
            }
        };
        let _ = writeln!(err, "[timing] search: {}", fmt(timings.reranking));
        let _ = writeln!(err, "[timing] augmentation: {}", fmt(timings.augmentation));
        let _ = writeln!(err, "[timing] total: {}", fmt(timings.total));
    }

    Ok(())
}
