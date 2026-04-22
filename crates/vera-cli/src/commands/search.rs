//! `vera search <query>` — Search the indexed codebase.

use anyhow::bail;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use vera_core::config::InferenceBackend;
use vera_core::retrieval::search_service::SearchTimings;
use vera_core::types::SearchResult;

use crate::helpers::{load_runtime_config, output_results, prepare_indexed_search};

/// Run the `vera search <query>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    queries: &[String],
    intent: Option<&str>,
    limit: Option<usize>,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
    timing: bool,
    deep: bool,
    git_scope: Option<vera_core::git_scope::GitScope>,
    compact: bool,
    backend: InferenceBackend,
) -> anyhow::Result<()> {
    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);
    let queries = normalize_queries(queries);

    if queries.is_empty() {
        bail!(
            "search query is empty.\n\
             Hint: pass at least one non-empty quoted query."
        );
    }

    let (index_dir, filters) =
        prepare_indexed_search(&config.indexing, filters, git_scope.as_ref())?;

    let (results, timings) = if queries.len() == 1 {
        let effective_query = apply_intent(&queries[0], intent);
        execute_query(
            &index_dir,
            &effective_query,
            &config,
            &filters,
            result_limit,
            backend,
            deep,
        )?
    } else {
        execute_multi_query_search(
            &index_dir,
            &queries,
            intent,
            &config,
            &filters,
            result_limit,
            backend,
            deep,
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
        print_timings(&timings);
    }

    Ok(())
}

fn execute_query(
    index_dir: &Path,
    query: &str,
    config: &vera_core::config::VeraConfig,
    filters: &vera_core::types::SearchFilters,
    result_limit: usize,
    backend: InferenceBackend,
    deep: bool,
) -> anyhow::Result<(Vec<SearchResult>, SearchTimings)> {
    if deep {
        vera_core::retrieval::rag_fusion::execute_deep_search(
            index_dir,
            query,
            config,
            filters,
            result_limit,
            backend,
        )
    } else {
        vera_core::retrieval::search_service::execute_search(
            index_dir,
            query,
            config,
            filters,
            result_limit,
            backend,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_multi_query_search(
    index_dir: &Path,
    queries: &[String],
    intent: Option<&str>,
    config: &vera_core::config::VeraConfig,
    filters: &vera_core::types::SearchFilters,
    result_limit: usize,
    backend: InferenceBackend,
    deep: bool,
) -> anyhow::Result<(Vec<SearchResult>, SearchTimings)> {
    let overall_start = Instant::now();
    let per_query_limit = compute_per_query_limit(result_limit);
    let mut timings = SearchTimings::default();
    let mut weights = Vec::with_capacity(queries.len());
    let mut result_sets = Vec::with_capacity(queries.len());

    for query in queries {
        let effective_query = apply_intent(query, intent);
        let (results, query_timings) = execute_query(
            index_dir,
            &effective_query,
            config,
            filters,
            per_query_limit,
            backend,
            deep,
        )?;
        merge_timings(&mut timings, &query_timings);
        result_sets.push(results);
        weights.push(1.0);
    }

    let slices: Vec<&[SearchResult]> = result_sets.iter().map(Vec::as_slice).collect();
    let fused = vera_core::retrieval::fuse_rrf_multi_weighted(
        &slices,
        &weights,
        config.retrieval.rrf_k,
        result_limit,
    );
    let fused = vera_core::retrieval::search_service::augment_multi_query_exact_matches(
        index_dir,
        queries,
        fused,
        filters,
        result_limit,
    )?;
    timings.total = Some(overall_start.elapsed());
    Ok((fused, timings))
}

fn normalize_queries(queries: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(queries.len());
    let mut seen = std::collections::HashSet::new();

    for query in queries {
        let collapsed = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            continue;
        }
        if seen.insert(collapsed.to_ascii_lowercase()) {
            normalized.push(collapsed);
        }
    }

    normalized
}

fn apply_intent(query: &str, intent: Option<&str>) -> String {
    let intent = intent
        .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|value| !value.is_empty());
    match intent {
        Some(intent) => format!("intent: {intent} | {query}"),
        None => query.to_string(),
    }
}

fn compute_per_query_limit(result_limit: usize) -> usize {
    result_limit
        .saturating_mul(2)
        .max(result_limit.saturating_add(10))
        .max(20)
}

fn merge_timings(target: &mut SearchTimings, incoming: &SearchTimings) {
    add_duration(&mut target.embedding, incoming.embedding);
    add_duration(&mut target.bm25, incoming.bm25);
    add_duration(&mut target.vector, incoming.vector);
    add_duration(&mut target.fusion, incoming.fusion);
    add_duration(&mut target.reranking, incoming.reranking);
    add_duration(&mut target.augmentation, incoming.augmentation);
}

fn add_duration(target: &mut Option<Duration>, incoming: Option<Duration>) {
    if let Some(incoming) = incoming {
        *target = Some(target.unwrap_or_default() + incoming);
    }
}

fn print_timings(timings: &SearchTimings) {
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
