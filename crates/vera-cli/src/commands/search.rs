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
    deep: bool,
    compact: bool,
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

    let (results, timings) = if deep {
        vera_core::retrieval::rag_fusion::execute_deep_search(
            &index_dir,
            query,
            &config,
            filters,
            result_limit,
            backend,
        )?
    } else {
        vera_core::retrieval::search_service::execute_search(
            &index_dir,
            query,
            &config,
            filters,
            result_limit,
            backend,
        )?
    };

    output_results(
        &results,
        json_output,
        raw,
        compact,
        config.retrieval.max_output_chars,
    );

    if timing {
        use std::io::Write;
        use std::time::Duration;
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        let fmt = |d: Option<Duration>| -> String {
            match d {
                Some(d) => format!("{}ms", d.as_millis()),
                None => "n/a".to_string(),
            }
        };
        let stages: &[(&str, Option<Duration>)] = &[
            ("embedding", timings.embedding),
            ("bm25", timings.bm25),
            ("vector", timings.vector),
            ("fusion", timings.fusion),
            ("reranking", timings.reranking),
            ("augmentation", timings.augmentation),
            ("total", timings.total),
        ];
        for (name, duration) in stages {
            if duration.is_some() || *name == "total" {
                let _ = writeln!(err, "[timing] {name}: {}", fmt(*duration));
            }
        }
    }

    Ok(())
}
