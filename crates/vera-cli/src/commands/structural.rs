//! `vera structural` — agent-oriented structural search intents.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use clap::ValueEnum;

use crate::helpers::{load_runtime_config, output_results, warn_if_index_stale};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum StructuralIntent {
    #[value(alias = "defs")]
    Definitions,
    Calls,
    Env,
    Routes,
    Sql,
    #[value(alias = "impl")]
    Impls,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    intent: StructuralIntent,
    query: Option<&str>,
    limit: Option<usize>,
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

    let (kind, query) = match intent {
        StructuralIntent::Definitions => (
            vera_core::retrieval::StructuralSearchKind::Definitions,
            query,
        ),
        StructuralIntent::Calls => (vera_core::retrieval::StructuralSearchKind::Calls, query),
        StructuralIntent::Env => (vera_core::retrieval::StructuralSearchKind::EnvReads, query),
        StructuralIntent::Routes => (
            vera_core::retrieval::StructuralSearchKind::RouteHandlers,
            None,
        ),
        StructuralIntent::Sql => (vera_core::retrieval::StructuralSearchKind::SqlQueries, None),
        StructuralIntent::Impls => (
            vera_core::retrieval::StructuralSearchKind::Implementations,
            query,
        ),
    };

    let started_at = Instant::now();
    let results = vera_core::retrieval::search_structural(
        &index_dir,
        kind,
        query,
        limit.unwrap_or(20),
        &filters,
    )?;

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
