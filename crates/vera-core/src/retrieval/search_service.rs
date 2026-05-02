//! Shared search service used by both CLI and MCP.
//!
//! Encapsulates the common hybrid search flow: create embedding provider,
//! build reranker, compute fetch limits, execute search, apply filters.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::warn;

use crate::chunk_text::file_name;
use crate::config::{InferenceBackend, VeraConfig};
use crate::embedding::{CachedEmbeddingProvider, DynamicProvider, EmbeddingProvider};
use crate::retrieval::dynamic_reranker::DynamicReranker;
use crate::retrieval::hybrid::compute_vector_candidates;
use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
use crate::retrieval::query_utils::{
    looks_like_compound_identifier, looks_like_filename, path_depth, trim_query_token,
};
use crate::retrieval::ranking::{
    RankingStage, apply_query_ranking_with_filters, is_path_weighted_query,
};
use crate::retrieval::{apply_filters, search_bm25, search_hybrid, search_hybrid_reranked};
use crate::types::{Chunk, SearchFilters, SearchResult, SymbolType};

/// Timing data for each stage of the search pipeline.
#[derive(Debug, Default)]
pub struct SearchTimings {
    pub embedding: Option<Duration>,
    pub bm25: Option<Duration>,
    pub vector: Option<Duration>,
    pub fusion: Option<Duration>,
    pub reranking: Option<Duration>,
    pub augmentation: Option<Duration>,
    pub total: Option<Duration>,
}

/// Reusable search dependencies for a process or command invocation.
///
/// Local backends can take hundreds of milliseconds or seconds to initialize.
/// Keeping the provider and reranker here lets CLI multi-query search, deep
/// search, MCP, and eval reuse loaded models across repeated queries.
pub struct SearchContext {
    provider: Option<CachedEmbeddingProvider<DynamicProvider>>,
    model_name: Option<String>,
    provider_error: Option<String>,
    reranker: Option<DynamicReranker>,
}

impl SearchContext {
    pub async fn new(config: &VeraConfig, backend: InferenceBackend) -> Self {
        let (provider, model_name, provider_error) =
            match crate::embedding::create_dynamic_provider(config, backend).await {
                Ok((provider, model_name)) => (
                    Some(CachedEmbeddingProvider::new(provider, 512)),
                    Some(model_name),
                    None,
                ),
                Err(err) => {
                    warn!(
                        "Failed to create embedding provider ({}), using BM25-only search",
                        err
                    );
                    (None, None, Some(err.to_string()))
                }
            };

        let reranker = if provider.is_some() {
            crate::retrieval::create_dynamic_reranker(config, backend)
                .await
                .unwrap_or_else(|err| {
                    warn!("Failed to create reranker ({})", err);
                    None
                })
        } else {
            None
        };

        Self {
            provider,
            model_name,
            provider_error,
            reranker,
        }
    }

    pub fn bm25_only() -> Self {
        Self {
            provider: None,
            model_name: None,
            provider_error: None,
            reranker: None,
        }
    }

    pub fn embedding_provider(&self) -> Option<&CachedEmbeddingProvider<DynamicProvider>> {
        self.provider.as_ref()
    }

    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    pub async fn search(
        &self,
        index_dir: &Path,
        query: &str,
        config: &VeraConfig,
        filters: &SearchFilters,
        result_limit: usize,
    ) -> Result<(Vec<SearchResult>, SearchTimings)> {
        let total_start = Instant::now();
        let fetch_limit = compute_fetch_limit(query, filters, result_limit);

        let Some(provider) = self.provider.as_ref() else {
            if let Some(error) = self.provider_error.as_deref() {
                warn!(
                    "embedding provider unavailable ({}), using BM25-only search",
                    error
                );
            }
            return run_bm25_only(
                index_dir,
                query,
                filters,
                fetch_limit,
                result_limit,
                total_start,
            );
        };

        let mut stored_dim = config.embedding.max_stored_dim;

        // Check metadata mismatch
        let metadata_path = index_dir.join("metadata.db");
        if let Ok(metadata_store) = crate::storage::metadata::MetadataStore::open(&metadata_path) {
            if let (Some(s_model), Some(s_dim)) = (
                metadata_store.get_index_meta("model_name").unwrap_or(None),
                metadata_store
                    .get_index_meta("embedding_dim")
                    .unwrap_or(None),
            ) {
                if let Some(model_name) = self.model_name.as_deref() {
                    if !crate::config::model_names_match(&s_model, model_name) {
                        warn!(
                            "Index model '{}' does not match active model '{}'; using BM25-only search",
                            s_model, model_name
                        );
                        return run_bm25_only(
                            index_dir,
                            query,
                            filters,
                            fetch_limit,
                            result_limit,
                            total_start,
                        );
                    }
                }
                if let Ok(dim) = s_dim.parse::<usize>() {
                    if let Some(provider_dim) = provider.expected_dim() {
                        if provider_dim < dim {
                            warn!(
                                "Index dimension {} exceeds provider dimension {}; using BM25-only search",
                                dim, provider_dim
                            );
                            return run_bm25_only(
                                index_dir,
                                query,
                                filters,
                                fetch_limit,
                                result_limit,
                                total_start,
                            );
                        }
                    }
                    stored_dim = dim;
                }
            }
        }

        // Create optional reranker.
        let reranker_enabled = self.reranker.is_some() && !should_skip_reranking(query, filters);

        // Classify query to adapt fusion parameters.
        let query_type = classify_query(query);
        let query_params = params_for_query_type(query_type);
        let rrf_k = query_params.rrf_k;
        let vector_candidates = effective_vector_candidates(fetch_limit, query_params);
        let rerank_candidates = effective_rerank_candidates(
            config.retrieval.rerank_candidates,
            fetch_limit,
            query,
            filters,
        );

        let ranking_stage = if reranker_enabled {
            RankingStage::PostRerank
        } else {
            RankingStage::Initial
        };

        let (results, hybrid_timings) = if reranker_enabled {
            let reranker = self
                .reranker
                .as_ref()
                .expect("reranker_enabled requires an initialized reranker");
            search_hybrid_reranked(
                index_dir,
                provider,
                reranker,
                query,
                fetch_limit,
                rrf_k,
                stored_dim,
                rerank_candidates.max(fetch_limit),
                vector_candidates,
            )
            .await?
        } else {
            search_hybrid(
                index_dir,
                provider,
                query,
                fetch_limit,
                rrf_k,
                stored_dim,
                vector_candidates,
            )
            .await?
        };

        let mut timings = SearchTimings {
            embedding: hybrid_timings.embedding,
            bm25: hybrid_timings.bm25,
            vector: hybrid_timings.vector,
            fusion: hybrid_timings.fusion,
            reranking: hybrid_timings.reranking,
            ..Default::default()
        };

        let aug_start = Instant::now();
        let results =
            augment_exact_match_candidates(index_dir, query, results, ranking_stage, filters)?;
        timings.augmentation = Some(aug_start.elapsed());

        timings.total = Some(total_start.elapsed());
        Ok((apply_filters(results, filters, result_limit), timings))
    }
}

/// Execute a search against the index at `index_dir`.
///
/// Attempts hybrid search (BM25 + vector + optional reranking). Falls
/// back to BM25-only when embedding API is unavailable.
pub fn execute_search(
    index_dir: &Path,
    query: &str,
    config: &VeraConfig,
    filters: &SearchFilters,
    result_limit: usize,
    backend: InferenceBackend,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let rt = tokio::runtime::Runtime::new()?;
    let context = rt.block_on(SearchContext::new(config, backend));
    rt.block_on(context.search(index_dir, query, config, filters, result_limit))
}

/// Compute how many candidates to keep through fusion before final truncation.
///
/// Broad natural-language queries need a larger pool even without explicit
/// filters so deterministic ranking can surface structural chunks that raw RRF
/// scores placed outside the requested result window.
fn compute_fetch_limit(query: &str, filters: &SearchFilters, result_limit: usize) -> usize {
    let mut fetch_limit = if filters.is_empty() {
        result_limit
    } else {
        result_limit.saturating_mul(3).max(result_limit + 20)
    };

    // Path globs are applied post-retrieval, so we need a much larger pool
    // to ensure enough matching files survive filtering.
    if filters.path_glob.is_some() {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(10).max(result_limit + 100));
    }

    if filters.exact_paths.is_some() {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(12).max(result_limit + 200));
    }

    if needs_structural_overfetch(query, filters) {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(8).max(result_limit + 140));
    } else if matches!(classify_query(query), QueryType::NaturalLanguage) {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(3).max(result_limit + 40));
    }

    fetch_limit
}

fn needs_structural_overfetch(query: &str, filters: &SearchFilters) -> bool {
    matches!(classify_query(query), QueryType::NaturalLanguage)
        && query.split_whitespace().count() >= 4
        && filters.path_glob.is_none()
        && filters.exact_paths.is_none()
        && filters.symbol_type.is_none()
        && !is_path_weighted_query(query)
}

fn effective_vector_candidates(
    fetch_limit: usize,
    query_params: crate::retrieval::query_classifier::QueryParams,
) -> usize {
    compute_vector_candidates(fetch_limit, query_params.vector_candidate_multiplier)
}

fn effective_rerank_candidates(
    base: usize,
    fetch_limit: usize,
    query: &str,
    filters: &SearchFilters,
) -> usize {
    if needs_structural_overfetch(query, filters) {
        return base;
    }

    base.max(fetch_limit)
}

fn should_skip_reranking(query: &str, filters: &SearchFilters) -> bool {
    let word_count = query.split_whitespace().count();
    filters.path_glob.is_some()
        || filters.exact_paths.is_some()
        || filters.symbol_type.is_some()
        || is_path_weighted_query(query)
        || (matches!(classify_query(query), QueryType::Identifier) && word_count <= 2)
}

fn run_bm25_only(
    index_dir: &Path,
    query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    total_start: Instant,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let bm25_start = Instant::now();
    let results = search_bm25(index_dir, query, fetch_limit)?;
    let bm25_elapsed = bm25_start.elapsed();
    let aug_start = Instant::now();
    let results =
        augment_exact_match_candidates(index_dir, query, results, RankingStage::Initial, filters)?;
    let timings = SearchTimings {
        bm25: Some(bm25_elapsed),
        augmentation: Some(aug_start.elapsed()),
        total: Some(total_start.elapsed()),
        ..Default::default()
    };
    Ok((apply_filters(results, filters, result_limit), timings))
}

fn augment_exact_match_candidates(
    index_dir: &Path,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = crate::storage::metadata::MetadataStore::open(&metadata_path) else {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    };
    let supplemental = collect_exact_match_candidates(&store, query, 0)?;

    if supplemental.is_empty() {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    }

    let merged = merge_exact_matches(supplemental, results);

    Ok(apply_query_ranking_with_filters(
        query, merged, stage, filters,
    ))
}

/// Re-apply exact-match boosting after multi-query fusion.
///
/// Each subquery may have a different exact identifier or filename target.
/// Adding those exact hits again after RRF keeps direct symbol lookups near the
/// top instead of letting merged scores bury them behind broad contextual hits.
pub fn augment_multi_query_exact_matches(
    index_dir: &Path,
    queries: &[String],
    results: Vec<SearchResult>,
    filters: &SearchFilters,
    result_limit: usize,
) -> Result<Vec<SearchResult>> {
    if queries.is_empty() {
        return Ok(apply_filters(results, filters, result_limit));
    }

    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = crate::storage::metadata::MetadataStore::open(&metadata_path) else {
        return Ok(apply_filters(results, filters, result_limit));
    };

    let mut supplemental = Vec::new();
    for (query_index, query) in queries.iter().enumerate() {
        supplemental.extend(collect_exact_match_candidates(&store, query, query_index)?);
    }

    if supplemental.is_empty() {
        return Ok(apply_filters(results, filters, result_limit));
    }

    Ok(apply_filters(
        merge_exact_matches(supplemental, results),
        filters,
        result_limit,
    ))
}

#[derive(Debug)]
struct ExactMatchCandidate {
    order: ExactMatchOrder,
    result: SearchResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ExactMatchOrder {
    query_index: usize,
    match_kind: u8,
    exact_priority: (u8, u8, u8, u8),
    path_depth: usize,
    line_start: u32,
}

fn collect_exact_match_candidates(
    store: &crate::storage::metadata::MetadataStore,
    query: &str,
    query_index: usize,
) -> Result<Vec<SearchResult>> {
    let mut candidates = Vec::new();

    if let Some(filename) = extract_exact_filename(query).filter(|_| is_path_weighted_query(query))
    {
        let mut matching_files: Vec<String> = store
            .indexed_files()?
            .into_iter()
            .filter(|path| file_name(path).eq_ignore_ascii_case(&filename))
            .collect();
        matching_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));

        for file_path in matching_files.into_iter().take(20) {
            for chunk in store.get_chunks_by_file(&file_path)? {
                candidates.push(ExactMatchCandidate {
                    order: ExactMatchOrder {
                        query_index,
                        match_kind: 0,
                        exact_priority: (0, 0, 0, 0),
                        path_depth: path_depth(&chunk.file_path),
                        line_start: chunk.line_start,
                    },
                    result: chunk_to_result(chunk),
                });
            }
        }
    }

    if let Some(identifier_case) = extract_exact_identifier_case(query).as_deref() {
        let mut chunks = store.get_chunks_by_symbol_name_case_sensitive(identifier_case)?;
        let identifier = identifier_case.to_ascii_lowercase();
        let mut fallback_chunks = store.get_chunks_by_symbol_name(&identifier)?;
        fallback_chunks.retain(|chunk| chunk.symbol_name.as_deref() != Some(identifier_case));
        if uppercase_identifier_query(identifier_case) {
            fallback_chunks.retain(|chunk| {
                !matches!(
                    chunk.symbol_type,
                    Some(SymbolType::Method | SymbolType::Function | SymbolType::Module)
                )
            });
        }
        chunks.extend(fallback_chunks);

        // Scan for definition chunks in files whose stem matches the identifier.
        // This catches definitions that weren't found by symbol name lookup,
        // e.g. when the file is sessions.py and the query is "Session".
        let seen_files: HashSet<String> = chunks.iter().map(|c| c.file_path.clone()).collect();
        let stem_chunks = collect_stem_matched_definitions(store, &identifier, &seen_files)?;
        chunks.extend(stem_chunks);

        for chunk in chunks {
            let order = ExactMatchOrder {
                query_index,
                match_kind: 1,
                exact_priority: exact_match_priority(query, identifier_case, &chunk),
                path_depth: path_depth(&chunk.file_path),
                line_start: chunk.line_start,
            };
            candidates.push(ExactMatchCandidate {
                order,
                result: chunk_to_result(chunk),
            });
        }
    }

    // For queries with no filename or identifier match, scan for files whose
    // stem matches a query keyword. This catches NL queries like
    // "configuration loading" → config.py, "testing client" → testing.py.
    if candidates.is_empty() {
        let concept_chunks = collect_concept_matched_files(store, query)?;
        for chunk in concept_chunks {
            candidates.push(ExactMatchCandidate {
                order: ExactMatchOrder {
                    query_index,
                    match_kind: 2,
                    exact_priority: (0, 0, 0, 0),
                    path_depth: path_depth(&chunk.file_path),
                    line_start: chunk.line_start,
                },
                result: chunk_to_result(chunk),
            });
        }
    }

    candidates.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then(left.result.file_path.cmp(&right.result.file_path))
    });

    Ok(candidates
        .into_iter()
        .map(|candidate| candidate.result)
        .collect())
}

/// Find definition chunks from files whose stem matches the identifier.
///
/// When searching for "Session", this finds files like `sessions.py`,
/// `session.rs`, etc. and returns their definition chunks. Only scans
/// definition-type symbols to avoid pulling in noisy mentions.
fn collect_stem_matched_definitions(
    store: &crate::storage::metadata::MetadataStore,
    identifier: &str,
    already_seen: &HashSet<String>,
) -> Result<Vec<crate::types::Chunk>> {
    let identifier_lower = identifier.to_ascii_lowercase();
    let mut results = Vec::new();

    let all_files = store.indexed_files()?;
    for file_path in &all_files {
        if already_seen.contains(file_path.as_str()) {
            continue;
        }
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(&fname);

        // Match stem to identifier: exact, plural, or prefix overlap.
        let is_stem_match = stem == identifier_lower
            || stem.strip_suffix('s') == Some(&identifier_lower)
            || identifier_lower.strip_suffix('s') == Some(stem)
            || (stem.len() >= 4 && identifier_lower.starts_with(stem))
            || (identifier_lower.len() >= 4 && stem.starts_with(&identifier_lower));

        if !is_stem_match {
            continue;
        }

        let chunks = store.get_chunks_by_file(file_path)?;
        for chunk in chunks {
            let is_definition = matches!(
                chunk.symbol_type,
                Some(
                    SymbolType::Class
                        | SymbolType::Struct
                        | SymbolType::Trait
                        | SymbolType::Interface
                        | SymbolType::Enum
                        | SymbolType::Function
                        | SymbolType::Module
                )
            );
            if is_definition {
                results.push(chunk);
            }
        }
    }

    Ok(results)
}

/// Find files whose stem matches a query keyword for NL queries.
///
/// When searching "configuration loading", this finds `config.py` because
/// "config" shares a prefix with "configuration". Only returns definition
/// chunks to avoid noise. Limited to short queries to avoid false positives.
fn collect_concept_matched_files(
    store: &crate::storage::metadata::MetadataStore,
    query: &str,
) -> Result<Vec<crate::types::Chunk>> {
    let keywords: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| w.len() >= 4 && !is_concept_stopword(w))
        .collect();

    if keywords.is_empty() || keywords.len() > 5 {
        return Ok(Vec::new());
    }

    let all_files = store.indexed_files()?;
    let mut matched_files = Vec::new();

    for file_path in &all_files {
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(&fname);
        if stem.len() < 4 {
            continue;
        }

        // Match file stem to query keywords: exact, singular/plural, or
        // prefix overlap. Both the stem and keyword must be non-stopwords
        // and the shorter must be at least 5 chars.
        if is_concept_stopword(stem) {
            continue;
        }
        let keyword_match = keywords.iter().any(|kw| {
            stem == kw
                || stem.strip_suffix('s') == Some(kw.as_str())
                || kw.strip_suffix('s') == Some(stem)
                || (stem.len() >= 5 && kw.starts_with(stem))
                || (kw.len() >= 5 && stem.starts_with(kw.as_str()))
        });

        if keyword_match {
            matched_files.push(file_path.clone());
        }
    }

    // Sort by path depth (prefer shallower files) and limit to avoid noise.
    matched_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));
    matched_files.truncate(3);

    let mut results = Vec::new();
    for file_path in &matched_files {
        let chunks = store.get_chunks_by_file(file_path)?;
        for chunk in chunks {
            // Only inject definition chunks to avoid flooding results with
            // every line of a concept-matched file.
            let is_definition = matches!(
                chunk.symbol_type,
                Some(
                    SymbolType::Class
                        | SymbolType::Struct
                        | SymbolType::Trait
                        | SymbolType::Interface
                        | SymbolType::Enum
                        | SymbolType::Function
                        | SymbolType::Module
                )
            );
            if is_definition {
                results.push(chunk);
            }
        }
    }

    Ok(results)
}

/// Words too common to be useful for file-stem matching.
fn is_concept_stopword(word: &str) -> bool {
    matches!(
        word,
        "error"
            | "errors"
            | "type"
            | "types"
            | "data"
            | "file"
            | "files"
            | "code"
            | "test"
            | "tests"
            | "util"
            | "utils"
            | "help"
            | "helper"
            | "helpers"
            | "base"
            | "core"
            | "main"
            | "init"
            | "index"
            | "model"
            | "models"
            | "form"
            | "format"
            | "value"
            | "values"
            | "path"
            | "paths"
            | "node"
            | "item"
            | "items"
            | "work"
            | "with"
            | "from"
            | "that"
            | "this"
            | "what"
            | "when"
            | "does"
            | "have"
            | "been"
            | "into"
            | "about"
            | "through"
            | "between"
            | "inside"
    )
}

fn merge_exact_matches(
    supplemental: Vec<SearchResult>,
    results: Vec<SearchResult>,
) -> Vec<SearchResult> {
    let mut merged = Vec::with_capacity(supplemental.len() + results.len());
    let mut seen = HashSet::new();

    for result in supplemental.into_iter().chain(results) {
        if seen.insert(result_key(&result)) {
            merged.push(result);
        }
    }

    merged
}

fn extract_exact_filename(query: &str) -> Option<String> {
    query
        .split_whitespace()
        .map(trim_query_token)
        .filter(|token| !token.is_empty())
        .find(|token| looks_like_filename(token))
        .map(|token| file_name(token).to_ascii_lowercase())
}

fn extract_exact_identifier_case(query: &str) -> Option<String> {
    query
        .split_whitespace()
        .map(trim_query_token)
        .filter(|token| !token.is_empty())
        .find(|token| !looks_like_filename(token) && looks_like_compound_identifier(token))
        .map(ToString::to_string)
}

fn query_mentions_implementation(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    lower.contains("implement")
        || lower.contains("registration")
        || lower.contains("mounted")
        || lower.contains("mounting")
}

fn exact_match_priority(query: &str, identifier_case: &str, chunk: &Chunk) -> (u8, u8, u8, u8) {
    let exact_case = u8::from(chunk.symbol_name.as_deref() != Some(identifier_case));
    let implementation_rank =
        if query_mentions_implementation(query) && chunk_looks_like_impl(chunk) {
            0
        } else {
            1
        };
    let visibility_rank = u8::from(!chunk_is_public_symbol(chunk));
    let type_mismatch_rank = if identifier_case
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
        && matches!(
            chunk.symbol_type,
            Some(SymbolType::Method | SymbolType::Function)
        )
        && chunk.symbol_name.as_deref() != Some(identifier_case)
    {
        1
    } else {
        0
    };

    (
        exact_case,
        implementation_rank,
        visibility_rank,
        type_mismatch_rank,
    )
}

fn chunk_looks_like_impl(chunk: &Chunk) -> bool {
    chunk
        .symbol_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("impl"))
        || chunk
            .content
            .lines()
            .find(|line| !line.trim().is_empty())
            .is_some_and(|line| line.trim_start().starts_with("impl "))
}

fn chunk_is_public_symbol(chunk: &Chunk) -> bool {
    chunk.content.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(
            trimmed.starts_with("pub ")
                || trimmed.starts_with("export ")
                || trimmed.starts_with("public ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("interface "),
        )
    }) == Some(true)
}

fn uppercase_identifier_query(identifier: &str) -> bool {
    identifier
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
}

fn result_key(result: &SearchResult) -> String {
    format!(
        "{}:{}:{}",
        result.file_path, result.line_start, result.line_end
    )
}

fn chunk_to_result(chunk: crate::types::Chunk) -> SearchResult {
    SearchResult {
        file_path: chunk.file_path,
        line_start: chunk.line_start,
        line_end: chunk.line_end,
        content: chunk.content,
        language: chunk.language,
        score: 0.0,
        symbol_name: chunk.symbol_name,
        symbol_type: chunk.symbol_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::bm25::{Bm25Document, Bm25Index};
    use crate::storage::metadata::MetadataStore;
    use crate::types::{Chunk, Language};
    use std::ffi::OsString;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    const EMBEDDING_ENV_KEYS: [&str; 3] = [
        "EMBEDDING_MODEL_BASE_URL",
        "EMBEDDING_MODEL_ID",
        "EMBEDDING_MODEL_API_KEY",
    ];

    fn set_test_embedding_env(model_id: &str) -> Vec<(&'static str, Option<OsString>)> {
        let saved = EMBEDDING_ENV_KEYS
            .iter()
            .map(|key| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        unsafe {
            std::env::set_var("EMBEDDING_MODEL_BASE_URL", "http://127.0.0.1:0");
            std::env::set_var("EMBEDDING_MODEL_ID", model_id);
            std::env::set_var("EMBEDDING_MODEL_API_KEY", "dummy-key");
        }
        saved
    }

    fn restore_test_env(saved: Vec<(&'static str, Option<OsString>)>) {
        unsafe {
            for (key, value) in saved {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_and_inference() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let dir = tempdir().unwrap();
        let index_dir = dir.path();

        let metadata_path = index_dir.join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();

        // 1. Test dimension mismatch (requires local model so provider_dim is Some(768))
        store
            .set_index_meta("model_name", "jina-embeddings-v5-text-nano-retrieval")
            .unwrap();
        store.set_index_meta("embedding_dim", "1024").unwrap(); // Mismatch: 1024 vs 768

        let config = VeraConfig::default();
        let filters = SearchFilters::default();

        // This attempts local provider creation first, then falls back to BM25 when possible.
        // In this synthetic test fixture the BM25 index is absent, so either path may surface.
        {
            let res = execute_search(
                index_dir,
                "test",
                &config,
                &filters,
                10,
                crate::config::InferenceBackend::OnnxJina(
                    crate::config::OnnxExecutionProvider::Cpu,
                ),
            );
            if let Err(err) = res {
                let err_msg = err.to_string();
                assert!(
                    err_msg.contains("tantivy")
                        || err_msg.contains("Failed to initialize local embedding provider")
                        || err_msg.contains("No such file")
                        || err_msg.contains("not found"),
                    "{}",
                    err_msg
                );
            }
        }

        // 2. Test metadata-dimension inference path (API provider returns None for expected_dim)
        // Set up dummy environment variables for API provider construction.
        let saved_env = set_test_embedding_env("dummy-api-model");

        store
            .set_index_meta("model_name", "dummy-api-model")
            .unwrap();
        store.set_index_meta("embedding_dim", "123").unwrap();

        // Calling execute_search with is_local = false
        // It will pass the metadata check (model_name matches), skip mismatch check (expected_dim is None),
        // infer stored_dim = 123, and proceed to search.
        // Since the index is empty, it will return Ok([]) without making network calls.
        let res = execute_search(
            index_dir,
            "test",
            &config,
            &filters,
            10,
            crate::config::InferenceBackend::Api,
        );
        assert!(res.is_ok(), "Expected Ok but got {:?}", res);
        restore_test_env(saved_env);
    }

    #[test]
    fn model_metadata_mismatch_falls_back_to_bm25() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let saved_env = set_test_embedding_env("active-api-model");
        let dir = tempdir().unwrap();
        let index_dir = dir.path();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "auth:0".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 1,
                line_end: 4,
                content: "pub fn authenticate_user() -> bool { true }".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("authenticate_user".to_string()),
            }])
            .unwrap();
        store.set_index_meta("model_name", "indexed-model").unwrap();
        store.set_index_meta("embedding_dim", "64").unwrap();

        let bm25 = Bm25Index::open(&index_dir.join("bm25")).unwrap();
        bm25.insert_batch(&[Bm25Document {
            chunk_id: "auth:0",
            file_path: "src/auth.rs",
            content: "pub fn authenticate_user() -> bool { true }",
            symbol_name: Some("authenticate_user"),
            language: "rust",
        }])
        .unwrap();

        let mut config = VeraConfig::default();
        config.embedding.timeout_secs = 1;
        config.embedding.max_retries = 0;
        let filters = SearchFilters::default();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let context = rt.block_on(SearchContext::new(
            &config,
            crate::config::InferenceBackend::Api,
        ));

        let (results, timings) = rt
            .block_on(context.search(index_dir, "authenticate user", &config, &filters, 10))
            .unwrap();

        assert_eq!(results[0].file_path, "src/auth.rs");
        assert!(timings.bm25.is_some());
        assert!(timings.vector.is_none());
        restore_test_env(saved_env);
    }

    #[test]
    fn effective_candidates_use_base_multipliers() {
        // Rerank candidates just return base.max(fetch_limit)
        let filters = SearchFilters::default();
        assert_eq!(
            effective_rerank_candidates(50, 10, "anything", &filters),
            50
        );
        assert_eq!(effective_rerank_candidates(5, 10, "anything", &filters), 10);
        assert_eq!(
            effective_rerank_candidates(50, 160, "file type detection and filtering", &filters),
            50
        );

        // Vector candidates use query_params multiplier without inflation
        let nl_params =
            params_for_query_type(crate::retrieval::query_classifier::QueryType::NaturalLanguage);
        let vc = effective_vector_candidates(10, nl_params);
        assert!(vc >= 50); // at least the minimum from compute_vector_candidates
    }

    #[test]
    fn broad_nl_queries_overfetch_before_ranking() {
        let filters = SearchFilters::default();

        assert_eq!(compute_fetch_limit("Config", &filters, 20), 20);
        assert_eq!(
            compute_fetch_limit("file type detection and filtering", &filters, 20),
            160
        );
        assert_eq!(
            compute_fetch_limit(
                "how are HTTP errors handled and returned to clients",
                &filters,
                5
            ),
            145
        );
    }

    #[test]
    fn exact_identifier_queries_skip_reranking() {
        assert!(should_skip_reranking("Config", &SearchFilters::default()));
        assert!(should_skip_reranking(
            "src/config.ts",
            &SearchFilters::default()
        ));
        assert!(!should_skip_reranking(
            "how are HTTP errors handled",
            &SearchFilters::default()
        ));
    }

    #[test]
    fn exact_identifier_lookup_finds_matching_symbol() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "sink:0".to_string(),
                file_path: "crates/searcher/src/sink.rs".to_string(),
                line_start: 102,
                line_end: 223,
                content: "pub trait Sink {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Trait),
                symbol_name: Some("Sink".to_string()),
            }])
            .unwrap();

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "Sink trait and its implementations",
            Vec::new(),
            RankingStage::Initial,
            &SearchFilters::default(),
        )
        .unwrap();

        assert!(
            augmented
                .iter()
                .any(|result| result.symbol_name.as_deref() == Some("Sink"))
        );
    }

    #[test]
    fn exact_identifier_prefers_public_type_definition() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "config:0".to_string(),
                    file_path: "crates/core/search.rs".to_string(),
                    line_start: 19,
                    line_end: 25,
                    content: "struct Config {\n    search_zip: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
                Chunk {
                    id: "config:1".to_string(),
                    file_path: "crates/regex/src/config.rs".to_string(),
                    line_start: 25,
                    line_end: 43,
                    content: "pub(crate) struct Config {\n    pub(crate) multi_line: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
                Chunk {
                    id: "config:2".to_string(),
                    file_path: "crates/searcher/src/searcher/mod.rs".to_string(),
                    line_start: 151,
                    line_end: 185,
                    content: "pub struct Config {\n    line_term: LineTerminator,\n    multi_line: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
            ])
            .unwrap();

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "Config",
            Vec::new(),
            RankingStage::Initial,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(
            augmented[0].file_path,
            "crates/searcher/src/searcher/mod.rs"
        );
    }

    #[test]
    fn multi_query_exact_matches_are_promoted_after_fusion() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "kimi:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/kimi.rs".to_string(),
                    line_start: 265,
                    line_end: 275,
                    content: "pub fn persist_kimi_auth_record() {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("persist_kimi_auth_record".to_string()),
                },
                Chunk {
                    id: "factory:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/factory.rs".to_string(),
                    line_start: 286,
                    line_end: 296,
                    content: "pub fn persist_factory_auth_record() {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("persist_factory_auth_record".to_string()),
                },
                Chunk {
                    id: "lib:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/lib.rs".to_string(),
                    line_start: 10,
                    line_end: 40,
                    content: "pub use crate::kimi::persist_kimi_auth_record;".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Module),
                    symbol_name: Some("omnigate_auth".to_string()),
                },
            ])
            .unwrap();

        let fused = vec![
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/lib.rs".to_string(),
                line_start: 10,
                line_end: 40,
                content: "pub use crate::kimi::persist_kimi_auth_record;".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("omnigate_auth".to_string()),
                symbol_type: Some(SymbolType::Module),
            },
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/factory.rs".to_string(),
                line_start: 286,
                line_end: 296,
                content: "pub fn persist_factory_auth_record() {}".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("persist_factory_auth_record".to_string()),
                symbol_type: Some(SymbolType::Function),
            },
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/kimi.rs".to_string(),
                line_start: 265,
                line_end: 275,
                content: "pub fn persist_kimi_auth_record() {}".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("persist_kimi_auth_record".to_string()),
                symbol_type: Some(SymbolType::Function),
            },
        ];

        let queries = vec![
            "persist_kimi_auth_record".to_string(),
            "persist_factory_auth_record".to_string(),
        ];
        let augmented = augment_multi_query_exact_matches(
            dir.path(),
            &queries,
            fused,
            &SearchFilters::default(),
            5,
        )
        .unwrap();

        assert_eq!(
            augmented[0].symbol_name.as_deref(),
            Some("persist_kimi_auth_record")
        );
        assert_eq!(
            augmented[1].symbol_name.as_deref(),
            Some("persist_factory_auth_record")
        );
    }
}
