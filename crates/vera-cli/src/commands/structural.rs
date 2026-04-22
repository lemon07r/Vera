//! `vera structural` — agent-oriented structural search intents.

use std::io::Write;
use std::time::Instant;

use clap::ValueEnum;

use crate::helpers::{load_runtime_config, output_results, prepare_indexed_search};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum StructuralIntent {
    #[value(alias = "defs")]
    Definitions,
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
    let (index_dir, filters) =
        prepare_indexed_search(&config.indexing, filters, git_scope.as_ref())?;

    let (kind, query) = match intent {
        StructuralIntent::Definitions => (
            vera_core::retrieval::StructuralSearchKind::Definitions,
            query,
        ),
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
