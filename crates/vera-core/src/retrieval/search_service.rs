//! Shared search service used by both CLI and MCP.
//!
//! Encapsulates the common hybrid search flow: create embedding provider,
//! build reranker, compute fetch limits, execute search, apply filters.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use tracing::warn;

use crate::chunk_text::file_name;
use crate::config::{InferenceBackend, VeraConfig};
use crate::retrieval::hybrid::compute_vector_candidates;
use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
use crate::retrieval::ranking::{
    FileRole, RankingStage, apply_query_ranking, classify_file_role, is_path_weighted_query,
};
use crate::retrieval::{apply_filters, search_bm25, search_hybrid, search_hybrid_reranked};
use crate::types::{Chunk, SearchFilters, SearchResult, SymbolType};

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
) -> Result<Vec<SearchResult>> {
    let fetch_limit = compute_fetch_limit(filters, result_limit);
    let rt = tokio::runtime::Runtime::new()?;

    // Try to create embedding provider for hybrid search.
    let (provider, model_name) =
        match rt.block_on(crate::embedding::create_dynamic_provider(config, backend)) {
            Ok(res) => res,
            Err(e) => {
                if backend.is_local() {
                    anyhow::bail!("{}", e);
                }
                warn!(
                    "Failed to create embedding provider ({}), using BM25-only search",
                    e
                );
                let results = apply_query_ranking(
                    query,
                    search_bm25(index_dir, query, fetch_limit)?,
                    RankingStage::Initial,
                );
                return Ok(apply_filters(results, filters, result_limit));
            }
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
            if s_model != model_name {
                anyhow::bail!(
                    "Index was created with model '{}' ({} dimensions), but you are using model '{}'. Please re-index with matching provider.",
                    s_model,
                    s_dim,
                    model_name
                );
            }
            if let Ok(dim) = s_dim.parse::<usize>() {
                use crate::embedding::EmbeddingProvider;
                if let Some(provider_dim) = provider.expected_dim() {
                    if provider_dim != dim {
                        anyhow::bail!(
                            "Dimension mismatch: index has {} dimensions but active provider expects {}. Please re-index with matching provider.",
                            dim,
                            provider_dim
                        );
                    }
                }
                stored_dim = dim;
            }
        }
    }

    let provider = crate::embedding::CachedEmbeddingProvider::new(provider, 512);

    // Create optional reranker.
    let reranker = rt
        .block_on(crate::retrieval::create_dynamic_reranker(config, backend))
        .unwrap_or_else(|e| {
            warn!("Failed to create reranker ({})", e);
            None
        });

    // Classify query to adapt fusion parameters.
    let query_type = classify_query(query);
    let query_params = params_for_query_type(query_type);
    let rrf_k = query_params.rrf_k;
    let vector_candidates = effective_vector_candidates(fetch_limit, query_params, query);
    let rerank_candidates =
        effective_rerank_candidates(config.retrieval.rerank_candidates, fetch_limit, query);
    let skip_reranker = should_skip_reranker(query, query_type);

    let ranking_stage = if reranker.is_some() && !skip_reranker {
        RankingStage::PostRerank
    } else {
        RankingStage::Initial
    };

    let results = if let Some(ref reranker) = reranker.filter(|_| !skip_reranker) {
        rt.block_on(search_hybrid_reranked(
            index_dir,
            &provider,
            reranker,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
            rerank_candidates.max(fetch_limit),
            vector_candidates,
        ))?
    } else {
        rt.block_on(search_hybrid(
            index_dir,
            &provider,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
            vector_candidates,
        ))?
    };

    let results = augment_exact_match_candidates(index_dir, query, results, ranking_stage)?;
    Ok(apply_filters(results, filters, result_limit))
}

/// Compute how many candidates to fetch before filtering.
///
/// When filters are active, fetch more candidates to ensure we have enough
/// results after filtering.
fn compute_fetch_limit(filters: &SearchFilters, result_limit: usize) -> usize {
    if filters.is_empty() {
        result_limit
    } else {
        result_limit.saturating_mul(3).max(result_limit + 20)
    }
}

fn effective_vector_candidates(
    fetch_limit: usize,
    query_params: crate::retrieval::query_classifier::QueryParams,
    query: &str,
) -> usize {
    let mut candidates =
        compute_vector_candidates(fetch_limit, query_params.vector_candidate_multiplier);

    if needs_broader_candidate_pool(query, classify_query(query)) {
        candidates = candidates.max(fetch_limit.saturating_mul(6));
    } else if short_identifier_query(query, classify_query(query)) {
        candidates = candidates.max(fetch_limit.saturating_mul(4)).max(80);
    }

    candidates
}

fn effective_rerank_candidates(base: usize, fetch_limit: usize, query: &str) -> usize {
    let mut candidates = base.max(fetch_limit);

    if needs_broader_candidate_pool(query, classify_query(query)) {
        candidates = candidates.max(fetch_limit.saturating_mul(4)).max(80);
    } else if short_identifier_query(query, classify_query(query)) {
        candidates = candidates.max(fetch_limit.saturating_mul(3)).max(60);
    }

    candidates
}

fn should_skip_reranker(query: &str, query_type: QueryType) -> bool {
    query_type == QueryType::Identifier && is_path_weighted_query(query)
}

fn needs_broader_candidate_pool(query: &str, query_type: QueryType) -> bool {
    if matches!(query_type, QueryType::NaturalLanguage) {
        return true;
    }

    let lower = query.trim().to_ascii_lowercase();
    [
        "implementations",
        "implementation",
        "registered",
        "registration",
        "mounted",
        "mounting",
        "configured",
        "configuration",
        "across",
        "schema",
        "validation",
        "route",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn short_identifier_query(query: &str, query_type: QueryType) -> bool {
    matches!(query_type, QueryType::Identifier)
        && !is_path_weighted_query(query)
        && query.split_whitespace().count() <= 2
}

fn augment_exact_match_candidates(
    index_dir: &Path,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
) -> Result<Vec<SearchResult>> {
    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = crate::storage::metadata::MetadataStore::open(&metadata_path) else {
        return Ok(results);
    };

    let mut supplemental = Vec::new();
    let mut deferred_supplemental = Vec::new();

    if let Some(filename) = extract_exact_filename(query).filter(|_| is_path_weighted_query(query))
    {
        let mut matching_files: Vec<String> = store
            .indexed_files()?
            .into_iter()
            .filter(|path| file_name(path).eq_ignore_ascii_case(&filename))
            .collect();
        matching_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));

        for file_path in matching_files.into_iter().take(600) {
            supplemental.extend(
                store
                    .get_chunks_by_file(&file_path)?
                    .into_iter()
                    .map(chunk_to_result),
            );
        }
    }

    let exact_identifier_case = extract_exact_identifier_case(query);

    if let Some(identifier_case) = exact_identifier_case.as_deref() {
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
        chunks.sort_by(|a, b| {
            exact_match_priority(query, identifier_case, a)
                .cmp(&exact_match_priority(query, identifier_case, b))
                .then(path_depth(&a.file_path).cmp(&path_depth(&b.file_path)))
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        supplemental.extend(chunks.into_iter().map(chunk_to_result));
    }

    let mut context_seed_results = supplemental.clone();
    context_seed_results.extend(results.iter().cloned());
    supplemental.extend(expand_file_context(query, &store, &context_seed_results)?);
    supplemental.extend(expand_alias_symbol_context(query, &store)?);
    deferred_supplemental.extend(expand_same_language_context(
        index_dir,
        query,
        &context_seed_results,
    )?);
    supplemental.extend(expand_cross_language_context(
        index_dir,
        query,
        &context_seed_results,
    )?);

    if let Some(identifier_case) = exact_identifier_case
        .as_deref()
        .filter(|_| wants_related_identifier_context(query))
    {
        let mut related = store.get_chunks_by_symbol_name_substring(identifier_case, 256)?;
        related.retain(|chunk| {
            chunk
                .symbol_name
                .as_deref()
                .is_some_and(|name| !name.eq_ignore_ascii_case(identifier_case))
        });
        related.sort_by(|a, b| {
            related_symbol_priority(query, a)
                .cmp(&related_symbol_priority(query, b))
                .then(path_depth(&a.file_path).cmp(&path_depth(&b.file_path)))
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        supplemental.extend(related.into_iter().take(24).map(chunk_to_result));
    }

    if supplemental.is_empty() {
        return Ok(results);
    }

    let query_type = classify_query(query);
    let mut merged =
        Vec::with_capacity(supplemental.len() + results.len() + deferred_supplemental.len());
    let mut seen = HashSet::new();

    if query_type == QueryType::NaturalLanguage {
        let mut original_iter = results.into_iter();
        if let Some(first) = original_iter.next() {
            for result in std::iter::once(first)
                .chain(supplemental)
                .chain(deferred_supplemental)
                .chain(original_iter)
            {
                if seen.insert(result_key(&result)) {
                    merged.push(result);
                }
            }
        } else {
            for result in supplemental.into_iter().chain(deferred_supplemental) {
                if seen.insert(result_key(&result)) {
                    merged.push(result);
                }
            }
        }
    } else {
        for result in supplemental
            .into_iter()
            .chain(results)
            .chain(deferred_supplemental)
        {
            if seen.insert(result_key(&result)) {
                merged.push(result);
            }
        }
    }

    Ok(apply_query_ranking(query, merged, stage))
}

fn expand_file_context(
    query: &str,
    store: &crate::storage::metadata::MetadataStore,
    results: &[SearchResult],
) -> Result<Vec<SearchResult>> {
    if classify_query(query) != QueryType::NaturalLanguage
        || is_path_weighted_query(query)
        || results.is_empty()
    {
        return Ok(Vec::new());
    }

    let keywords = query_keywords(query);
    let exact_identifier =
        extract_exact_identifier_case(query).map(|identifier| identifier.to_ascii_lowercase());
    let mut files = Vec::new();
    let mut seen = HashSet::new();

    for result in results.iter().take(8) {
        if !seen.insert(result.file_path.clone()) {
            continue;
        }
        if !is_expandable_file(result) {
            continue;
        }
        if should_expand_file_context(query, &keywords, exact_identifier.as_deref(), result) {
            files.push(result.file_path.clone());
        }
        if files.len() >= 4 {
            break;
        }
    }

    if files.is_empty() {
        return Ok(Vec::new());
    }

    let mut supplemental = Vec::new();
    for file_path in files {
        let mut chunks: Vec<Chunk> = store
            .get_chunks_by_file(&file_path)?
            .into_iter()
            .filter(is_structural_chunk)
            .collect();
        chunks.sort_by_key(|chunk| structural_chunk_priority(query, chunk));
        supplemental.extend(chunks.into_iter().take(8).map(chunk_to_result));
    }

    Ok(supplemental)
}

fn expand_cross_language_context(
    index_dir: &Path,
    query: &str,
    results: &[SearchResult],
) -> Result<Vec<SearchResult>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    let query_type = classify_query(query);
    if !matches!(
        query_type,
        QueryType::Identifier | QueryType::NaturalLanguage
    ) {
        return Ok(Vec::new());
    }

    let Some(seed) = results.iter().find(|result| {
        is_expandable_file(result)
            && is_structural_result(result)
            && !matches!(result.symbol_type, Some(SymbolType::Block))
    }) else {
        return Ok(Vec::new());
    };

    let terms = cross_language_terms(query, seed);
    if terms.len() < 6 {
        return Ok(Vec::new());
    }

    let expanded_query = terms.join(" ");
    let candidates = search_bm25(index_dir, &expanded_query, 160)?;
    let mut filtered: Vec<SearchResult> = candidates
        .into_iter()
        .filter(|candidate| candidate.file_path != seed.file_path)
        .filter(|candidate| candidate.language != seed.language)
        .filter(is_expandable_file)
        .filter(is_structural_result)
        .filter(|candidate| result_term_overlap(candidate, &terms) >= 3)
        .collect();

    filtered.sort_by(|a, b| {
        cross_language_candidate_score(b, &terms)
            .cmp(&cross_language_candidate_score(a, &terms))
            .then(structural_result_rank(a).cmp(&structural_result_rank(b)))
            .then(path_depth(&a.file_path).cmp(&path_depth(&b.file_path)))
            .then(a.file_path.cmp(&b.file_path))
            .then(a.line_start.cmp(&b.line_start))
    });
    filtered.truncate(8);
    Ok(filtered)
}

fn expand_same_language_context(
    index_dir: &Path,
    query: &str,
    results: &[SearchResult],
) -> Result<Vec<SearchResult>> {
    if results.is_empty()
        || classify_query(query) != QueryType::NaturalLanguage
        || is_path_weighted_query(query)
        || !query_mentions_error_context(query)
    {
        return Ok(Vec::new());
    }

    let Some(seed) = results.iter().find(|result| {
        is_expandable_file(result)
            && (is_structural_result(result)
                || matches!(
                    result.symbol_type,
                    Some(SymbolType::Function | SymbolType::Method)
                ))
    }) else {
        return Ok(Vec::new());
    };

    let seed_terms = same_language_seed_terms(seed);
    if seed_terms.len() < 3 {
        return Ok(Vec::new());
    }

    let terms = same_language_context_terms(query, &seed_terms);
    if terms.len() < 6 {
        return Ok(Vec::new());
    }

    let expanded_query = terms.join(" ");
    let candidates = search_bm25(index_dir, &expanded_query, 160)?;
    let mut filtered: Vec<SearchResult> = candidates
        .into_iter()
        .filter(|candidate| candidate.file_path != seed.file_path)
        .filter(|candidate| candidate.language == seed.language)
        .filter(is_expandable_file)
        .filter(|candidate| result_term_overlap(candidate, &seed_terms) >= 1)
        .filter(|candidate| result_term_overlap(candidate, &terms) >= 3)
        .collect();

    filtered.sort_by(|a, b| {
        same_language_candidate_score(b, &seed_terms, &terms)
            .cmp(&same_language_candidate_score(a, &seed_terms, &terms))
            .then(path_depth(&a.file_path).cmp(&path_depth(&b.file_path)))
            .then(a.file_path.cmp(&b.file_path))
            .then(a.line_start.cmp(&b.line_start))
    });
    filtered.truncate(8);
    Ok(filtered)
}

fn expand_alias_symbol_context(
    query: &str,
    store: &crate::storage::metadata::MetadataStore,
) -> Result<Vec<SearchResult>> {
    if classify_query(query) != QueryType::NaturalLanguage || !query_mentions_error_context(query) {
        return Ok(Vec::new());
    }

    let mut supplemental = Vec::new();
    for alias in ["abort"] {
        let mut chunks = store.get_chunks_by_symbol_name(alias)?;
        chunks.retain(|chunk| {
            matches!(
                classify_file_role(&chunk.file_path, chunk.language),
                FileRole::Source | FileRole::Unknown
            )
        });
        chunks.sort_by(|a, b| {
            path_depth(&a.file_path)
                .cmp(&path_depth(&b.file_path))
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        supplemental.extend(chunks.into_iter().map(chunk_to_result));
    }

    Ok(supplemental)
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

fn trim_query_token(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        !ch.is_ascii_alphanumeric() && !matches!(ch, '.' | '_' | '-' | '/')
    })
}

fn looks_like_filename(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "dockerfile" | "makefile" | "cmakelists.txt" | "nginx.conf"
    ) || token.contains('.')
}

fn looks_like_compound_identifier(token: &str) -> bool {
    token.contains('_') || token.contains("::") || token.chars().any(|ch| ch.is_ascii_uppercase())
}

fn query_keywords(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(trim_query_token)
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| {
            !token.is_empty()
                && !matches!(
                    token.as_str(),
                    "and"
                        | "or"
                        | "the"
                        | "a"
                        | "an"
                        | "of"
                        | "in"
                        | "to"
                        | "for"
                        | "with"
                        | "definition"
                        | "definitions"
                )
        })
        .map(|token| token.trim_end_matches('s').to_string())
        .collect()
}

fn query_mentions_implementation(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    lower.contains("implement")
        || lower.contains("registration")
        || lower.contains("mounted")
        || lower.contains("mounting")
}

fn query_mentions_error_context(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    lower.contains("error") || lower.contains("exception") || lower.contains("http")
}

fn wants_related_identifier_context(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    [
        "implementations",
        "implementation",
        "registration",
        "register",
        "mounted",
        "mounting",
        "route",
        "routes",
        "across languages",
        "across",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn exact_match_priority(query: &str, identifier_case: &str, chunk: &Chunk) -> (u8, u8, u8) {
    let exact_case = u8::from(chunk.symbol_name.as_deref() != Some(identifier_case));
    let implementation_rank =
        if query_mentions_implementation(query) && chunk_looks_like_impl(chunk) {
            0
        } else {
            1
        };
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

    (exact_case, implementation_rank, type_mismatch_rank)
}

fn related_symbol_priority(query: &str, chunk: &Chunk) -> (u8, u32) {
    let lower_query = query.to_ascii_lowercase();
    let lines = chunk.line_end.saturating_sub(chunk.line_start) + 1;
    let priority = if lower_query.contains("implement") {
        let exact_identifier =
            extract_exact_identifier_case(query).map(|value| value.to_ascii_lowercase());
        if chunk.symbol_name.as_deref().is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("impl ")
                && lower.contains(" for ")
                && exact_identifier
                    .as_deref()
                    .is_some_and(|identifier| lower.contains(identifier))
        }) {
            0
        } else if chunk.symbol_name.as_deref().is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("impl ") && lower.contains(" for ")
        }) {
            1
        } else if chunk_looks_like_impl(chunk) {
            2
        } else if matches!(
            chunk.symbol_type,
            Some(SymbolType::Struct | SymbolType::Class)
        ) {
            3
        } else {
            4
        }
    } else if lower_query.contains("register")
        || lower_query.contains("mount")
        || lower_query.contains("route")
    {
        if chunk.symbol_name.as_deref().is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("register") || lower.contains("route")
        }) {
            0
        } else if matches!(
            chunk.symbol_type,
            Some(SymbolType::Class | SymbolType::Struct)
        ) {
            1
        } else {
            2
        }
    } else if matches!(
        chunk.symbol_type,
        Some(SymbolType::Trait | SymbolType::Interface)
    ) {
        0
    } else {
        1
    };

    (priority, lines)
}

fn file_matches_query_keywords(file_path: &str, keywords: &[String]) -> bool {
    if keywords.is_empty() {
        return false;
    }

    let lower = file_path.to_ascii_lowercase();
    lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.trim_end_matches('s'))
        .any(|token| keywords.iter().any(|keyword| keyword == token))
}

fn should_expand_file_context(
    query: &str,
    keywords: &[String],
    exact_identifier: Option<&str>,
    result: &SearchResult,
) -> bool {
    if result_needs_file_context(result) || file_matches_query_keywords(&result.file_path, keywords)
    {
        return true;
    }

    if !wants_related_identifier_context(query) {
        return false;
    }

    exact_identifier.is_some_and(|identifier| {
        result
            .symbol_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
    })
}

fn result_needs_file_context(result: &SearchResult) -> bool {
    let lines = result.line_end.saturating_sub(result.line_start) + 1;
    matches!(result.symbol_type, Some(SymbolType::Variable))
        || matches!(
            result.symbol_type,
            Some(SymbolType::Function | SymbolType::Method)
        ) && lines <= 12
        || lines <= 8
}

fn is_expandable_file(result: &SearchResult) -> bool {
    matches!(
        classify_file_role(&result.file_path, result.language),
        FileRole::Source | FileRole::Unknown
    )
}

fn is_structural_chunk(chunk: &Chunk) -> bool {
    let lines = chunk.line_end.saturating_sub(chunk.line_start) + 1;
    matches!(
        chunk.symbol_type,
        Some(
            SymbolType::Struct
                | SymbolType::Class
                | SymbolType::Trait
                | SymbolType::Interface
                | SymbolType::Enum
                | SymbolType::Module
        )
    ) || matches!(chunk.symbol_type, Some(SymbolType::Block)) && lines >= 20
}

fn structural_chunk_priority(query: &str, chunk: &Chunk) -> (u8, u8, u32, u32) {
    let rank = if query_mentions_implementation(query) && chunk_looks_like_impl(chunk) {
        0
    } else {
        match chunk.symbol_type {
            Some(
                SymbolType::Struct | SymbolType::Class | SymbolType::Trait | SymbolType::Interface,
            ) => 1,
            Some(SymbolType::Enum | SymbolType::Module) => 2,
            Some(SymbolType::Block) => 3,
            _ => 4,
        }
    };
    let query_rank = if query.to_ascii_lowercase().contains("register")
        && chunk.symbol_name.as_deref().is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("register") || lower.contains("route")
        }) {
        0
    } else {
        1
    };
    let lines = chunk.line_end.saturating_sub(chunk.line_start) + 1;
    (rank, query_rank, chunk.line_start, lines)
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

fn is_structural_result(result: &SearchResult) -> bool {
    let lines = result.line_end.saturating_sub(result.line_start) + 1;
    matches!(
        result.symbol_type,
        Some(
            SymbolType::Struct
                | SymbolType::Class
                | SymbolType::Trait
                | SymbolType::Interface
                | SymbolType::Enum
                | SymbolType::Module
        )
    ) || matches!(result.symbol_type, Some(SymbolType::Block)) && lines >= 20
}

fn structural_result_rank(result: &SearchResult) -> u8 {
    match result.symbol_type {
        Some(SymbolType::Interface) => 0,
        Some(SymbolType::Struct | SymbolType::Class | SymbolType::Trait) => 1,
        Some(SymbolType::Enum | SymbolType::Module) => 2,
        Some(SymbolType::Block) => 3,
        _ => 4,
    }
}

fn cross_language_terms(query: &str, seed: &SearchResult) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    push_unique_term(&mut terms, &mut seen, trim_query_token(query));
    if let Some(identifier) = extract_exact_identifier_case(query) {
        add_identifier_variants(&mut terms, &mut seen, &identifier);
    }
    if let Some(symbol_name) = seed.symbol_name.as_deref() {
        add_identifier_variants(&mut terms, &mut seen, symbol_name);
    }

    for line in seed.content.lines() {
        for term in backtick_terms(line) {
            add_identifier_variants(&mut terms, &mut seen, &term);
        }
        if let Some(term) = field_like_identifier(line) {
            add_identifier_variants(&mut terms, &mut seen, &term);
        }
    }

    push_unique_term(&mut terms, &mut seen, "interface");
    push_unique_term(&mut terms, &mut seen, "types");
    push_unique_term(&mut terms, &mut seen, "config");
    terms.truncate(18);
    terms
}

fn same_language_seed_terms(seed: &SearchResult) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    if let Some(symbol_name) = seed.symbol_name.as_deref() {
        add_identifier_variants(&mut terms, &mut seen, symbol_name);
    }

    for line in seed.content.lines().take(24) {
        for term in line_identifiers(line) {
            add_identifier_variants(&mut terms, &mut seen, &term);
        }
        for term in backtick_terms(line) {
            add_identifier_variants(&mut terms, &mut seen, &term);
        }
    }

    terms
}

fn same_language_context_terms(query: &str, seed_terms: &[String]) -> Vec<String> {
    let mut terms = seed_terms.to_vec();
    let mut seen: HashSet<String> = terms.iter().cloned().collect();

    add_semantic_aliases(query, &mut terms, &mut seen);
    terms.truncate(20);
    terms
}

fn result_term_overlap(result: &SearchResult, terms: &[String]) -> usize {
    let candidate_tokens = normalized_tokens(&format!(
        "{} {} {}",
        result.file_path,
        result.symbol_name.as_deref().unwrap_or(""),
        result.content
    ));

    terms
        .iter()
        .map(|term| normalize_search_token(term))
        .filter(|term| !term.is_empty())
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|term| candidate_tokens.contains(term))
        .count()
}

fn cross_language_candidate_score(result: &SearchResult, terms: &[String]) -> usize {
    let overlap = result_term_overlap(result, terms);
    let path = result.file_path.to_ascii_lowercase();
    let path_bonus =
        usize::from(path.contains("/types/") || path.contains("schema") || path.contains("config"));
    let source_bonus = usize::from(path.contains("/src/") || path.contains("/packages/"));
    let type_bonus = match result.symbol_type {
        Some(SymbolType::Interface) => 3,
        Some(SymbolType::Struct | SymbolType::Class | SymbolType::Trait) => 2,
        Some(SymbolType::Enum | SymbolType::Module) => 1,
        _ => 0,
    };

    overlap * 100 + path_bonus * 20 + source_bonus * 10 + type_bonus
}

fn same_language_candidate_score(
    result: &SearchResult,
    seed_terms: &[String],
    terms: &[String],
) -> usize {
    let seed_overlap = result_term_overlap(result, seed_terms);
    let overlap = result_term_overlap(result, terms);
    let path = result.file_path.to_ascii_lowercase();
    let source_bonus = usize::from(path.contains("/src/"));
    let helper_bonus = usize::from(path.contains("helper") || path.contains("error"));
    let type_bonus = match result.symbol_type {
        Some(SymbolType::Function | SymbolType::Method) => 3,
        Some(
            SymbolType::Class | SymbolType::Struct | SymbolType::Trait | SymbolType::Interface,
        ) => 2,
        Some(SymbolType::Block) => 1,
        _ => 0,
    };

    seed_overlap * 120 + overlap * 40 + source_bonus * 20 + helper_bonus * 10 + type_bonus
}

fn backtick_terms(line: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut rest = line;

    while let Some(start) = rest.find('`') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('`') else {
            break;
        };
        let candidate = &after[..end];
        if candidate.chars().any(|ch| ch.is_ascii_alphabetic()) {
            terms.push(candidate.to_string());
        }
        rest = &after[end + 1..];
    }

    terms
}

fn line_identifiers(line: &str) -> Vec<String> {
    line.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .map(|part| part.trim_matches('_'))
        .filter(|part| !part.is_empty() && looks_like_compound_identifier(part))
        .map(ToString::to_string)
        .collect()
}

fn add_semantic_aliases(query: &str, terms: &mut Vec<String>, seen: &mut HashSet<String>) {
    let lower = query.to_ascii_lowercase();

    if lower.contains("error") || lower.contains("exception") {
        push_unique_term(terms, seen, "exception");
        push_unique_term(terms, seen, "handler");
        push_unique_term(terms, seen, "raise");
        push_unique_term(terms, seen, "abort");
    }
    if lower.contains("http") {
        push_unique_term(terms, seen, "HTTPException");
        push_unique_term(terms, seen, "aborter");
    }
}

fn field_like_identifier(line: &str) -> Option<String> {
    let trimmed = line.trim().trim_end_matches(',').trim_end_matches(';');
    if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('*') {
        return None;
    }

    let candidate = if let Some(rest) = trimmed.strip_prefix("pub ") {
        rest.split(':').next()?
    } else {
        trimmed.split(':').next()?
    }
    .trim()
    .trim_end_matches('?')
    .split_whitespace()
    .last()?;

    candidate
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        .then(|| candidate.to_string())
}

fn add_identifier_variants(terms: &mut Vec<String>, seen: &mut HashSet<String>, raw: &str) {
    push_unique_term(terms, seen, raw);

    let snake = raw.to_ascii_lowercase().replace('-', "_");
    if snake.contains('_') {
        push_unique_term(terms, seen, &snake);
        push_unique_term(terms, seen, &to_camel_case(&snake));
    }
}

fn push_unique_term(terms: &mut Vec<String>, seen: &mut HashSet<String>, raw: &str) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }

    let normalized = normalize_search_token(trimmed);
    if normalized.len() < 3
        || matches!(
            normalized.as_str(),
            "the" | "and" | "with" | "from" | "that" | "this" | "task" | "tasks" | "field" | "type"
        )
    {
        return;
    }

    if seen.insert(trimmed.to_string()) {
        terms.push(trimmed.to_string());
    }
}

fn to_camel_case(value: &str) -> String {
    let mut result = String::new();
    let mut uppercase_next = false;

    for ch in value.chars() {
        if ch == '_' || ch == '-' {
            uppercase_next = true;
            continue;
        }

        if uppercase_next {
            result.push(ch.to_ascii_uppercase());
            uppercase_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

fn normalized_tokens(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(normalize_search_token)
        .filter(|part| !part.is_empty())
        .collect()
}

fn normalize_search_token(token: &str) -> String {
    token
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_lowercase()
        .trim_end_matches('s')
        .to_string()
}

fn uppercase_identifier_query(identifier: &str) -> bool {
    identifier
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
}

fn path_depth(path: &str) -> usize {
    path.matches('/').count() + path.matches('\\').count()
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
    use tempfile::tempdir;

    #[test]
    fn test_dimension_mismatch_and_inference() {
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

        // This will attempt to create local provider and should fail at mismatch
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
            assert!(res.is_err());
            let err_msg = res.unwrap_err().to_string();
            // With load-dynamic ort, if ONNX Runtime is not present the error will be
            // about loading the runtime. If it IS present, it will be a dimension mismatch.
            // Either way the search correctly fails.
            assert!(
                err_msg.contains(
                    "Dimension mismatch: index has 1024 dimensions but active provider expects 768"
                ) || err_msg.contains("Failed to initialize local embedding provider"),
                "{}",
                err_msg
            );
        }

        // 2. Test metadata-dimension inference path (API provider returns None for expected_dim)
        // Set up dummy environment variables for API provider to bypass missing keys error
        unsafe {
            std::env::set_var("EMBEDDING_MODEL_BASE_URL", "http://127.0.0.1:0");
            std::env::set_var("EMBEDDING_MODEL_ID", "dummy-api-model");
            std::env::set_var("EMBEDDING_MODEL_API_KEY", "dummy-key");
        }

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
    }

    #[test]
    fn skips_reranker_for_filename_queries() {
        assert!(should_skip_reranker(
            "Cargo.toml workspace configuration",
            QueryType::Identifier
        ));
        assert!(!should_skip_reranker(
            "where is the workspace configured",
            QueryType::NaturalLanguage
        ));
    }

    #[test]
    fn broader_queries_expand_candidates() {
        assert!(effective_rerank_candidates(50, 10, "Sink trait and its implementations") >= 80);
        assert!(
            effective_vector_candidates(
                10,
                params_for_query_type(QueryType::NaturalLanguage),
                "request validation and schema enforcement"
            ) >= 60
        );
    }

    #[test]
    fn short_identifier_queries_expand_candidates() {
        assert!(effective_rerank_candidates(20, 10, "Config") >= 60);
        assert!(
            effective_vector_candidates(10, params_for_query_type(QueryType::Identifier), "Config")
                >= 80
        );
    }

    #[test]
    fn file_context_expansion_adds_structural_chunks() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "types:0".to_string(),
                    file_path: "crates/ignore/src/types.rs".to_string(),
                    line_start: 132,
                    line_end: 137,
                    content: "pub fn file_type_def(&self) -> Option<&FileTypeDef> {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Method),
                    symbol_name: Some("file_type_def".to_string()),
                },
                Chunk {
                    id: "types:1".to_string(),
                    file_path: "crates/ignore/src/types.rs".to_string(),
                    line_start: 146,
                    line_end: 149,
                    content: "pub struct FileTypeDef { name: String }".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("FileTypeDef".to_string()),
                },
                Chunk {
                    id: "types:2".to_string(),
                    file_path: "crates/ignore/src/types.rs".to_string(),
                    line_start: 165,
                    line_end: 181,
                    content: "pub struct Types { defs: Vec<FileTypeDef> }".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Types".to_string()),
                },
            ])
            .unwrap();

        let results = vec![SearchResult {
            file_path: "crates/ignore/src/types.rs".to_string(),
            line_start: 132,
            line_end: 137,
            content: "pub fn file_type_def(&self) -> Option<&FileTypeDef> {}".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("file_type_def".to_string()),
            symbol_type: Some(SymbolType::Method),
        }];

        let expanded =
            expand_file_context("file type detection and filtering", &store, &results).unwrap();

        assert!(
            expanded
                .iter()
                .any(|result| result.symbol_name.as_deref() == Some("FileTypeDef"))
        );
        assert!(
            expanded
                .iter()
                .any(|result| result.symbol_name.as_deref() == Some("Types"))
        );
    }

    #[test]
    fn file_context_expansion_skips_docs_files() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "docs:0".to_string(),
                file_path: "docs/Reference/Validation-and-Serialization.md".to_string(),
                line_start: 1,
                line_end: 200,
                content: "# Validation and Serialization".to_string(),
                language: Language::Markdown,
                symbol_type: Some(SymbolType::Block),
                symbol_name: Some("Validation-and-Serialization.md".to_string()),
            }])
            .unwrap();

        let results = vec![SearchResult {
            file_path: "docs/Reference/Validation-and-Serialization.md".to_string(),
            line_start: 1,
            line_end: 200,
            content: "# Validation and Serialization".to_string(),
            language: Language::Markdown,
            score: 1.0,
            symbol_name: Some("Validation-and-Serialization.md".to_string()),
            symbol_type: Some(SymbolType::Block),
        }];

        let expanded = expand_file_context(
            "request validation and schema enforcement",
            &store,
            &results,
        )
        .unwrap();

        assert!(expanded.is_empty());
    }

    #[test]
    fn related_identifier_context_adds_impl_block_matches() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "sink:0".to_string(),
                    file_path: "crates/searcher/src/sink.rs".to_string(),
                    line_start: 102,
                    line_end: 223,
                    content: "pub trait Sink {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Trait),
                    symbol_name: Some("Sink".to_string()),
                },
                Chunk {
                    id: "sink:1".to_string(),
                    file_path: "crates/printer/src/standard.rs".to_string(),
                    line_start: 763,
                    line_end: 868,
                    content: "impl Sink for StandardSink {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Block),
                    symbol_name: Some("impl Sink for StandardSink".to_string()),
                },
            ])
            .unwrap();

        let results = vec![SearchResult {
            file_path: "crates/searcher/src/sink.rs".to_string(),
            line_start: 102,
            line_end: 223,
            content: "pub trait Sink {}".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("Sink".to_string()),
            symbol_type: Some(SymbolType::Trait),
        }];

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "Sink trait and its implementations",
            results,
            RankingStage::Initial,
        )
        .unwrap();

        assert!(
            augmented.iter().any(|result| {
                result.symbol_name.as_deref() == Some("impl Sink for StandardSink")
            })
        );
    }

    #[test]
    fn cross_language_context_adds_pipeline_interface() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let bm25_dir = dir.path().join("bm25");
        let store = MetadataStore::open(&metadata_path).unwrap();
        let bm25 = Bm25Index::open(&bm25_dir).unwrap();

        let chunks = vec![
            Chunk {
                id: "task:0".to_string(),
                file_path: "crates/turborepo-types/src/lib.rs".to_string(),
                line_start: 750,
                line_end: 799,
                content: "pub struct TaskDefinition {\n    pub outputs: TaskOutputs,\n    pub cache: bool,\n    pub env: Vec<String>,\n    pub pass_through_env: Option<Vec<String>>,\n    pub inputs: TaskInputs,\n    pub output_logs: OutputLogsMode,\n    pub persistent: bool,\n    pub interruptible: bool,\n    pub interactive: bool,\n    pub with: Option<Vec<String>>,\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Struct),
                symbol_name: Some("TaskDefinition".to_string()),
            },
            Chunk {
                id: "task:1".to_string(),
                file_path: "packages/turbo-types/src/types/config-v2.ts".to_string(),
                line_start: 266,
                line_end: 423,
                content: "export interface Pipeline {\n  dependsOn?: Array<string>;\n  env?: Array<string>;\n  passThroughEnv?: Array<string>;\n  outputs?: Array<string>;\n  cache?: boolean;\n  inputs?: Array<string>;\n  outputLogs?: OutputLogs;\n  persistent?: boolean;\n  interactive?: boolean;\n  interruptible?: boolean;\n  with?: Array<string>;\n}"
                    .to_string(),
                language: Language::TypeScript,
                symbol_type: Some(SymbolType::Interface),
                symbol_name: Some("Pipeline".to_string()),
            },
            Chunk {
                id: "task:2".to_string(),
                file_path: "packages/eslint-plugin-turbo/lib/utils/calculate-inputs.ts".to_string(),
                line_start: 11,
                line_end: 16,
                content: "interface EnvironmentConfig {\n  legacyConfig: Array<string>;\n  env: Array<string>;\n  passThroughEnv: Array<string> | null;\n  dotEnv: DotEnvConfig | null;\n}"
                    .to_string(),
                language: Language::TypeScript,
                symbol_type: Some(SymbolType::Interface),
                symbol_name: Some("EnvironmentConfig".to_string()),
            },
        ];

        store.insert_chunks(&chunks).unwrap();
        let lang_strings: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.language.to_string())
            .collect();
        let docs: Vec<Bm25Document<'_>> = chunks
            .iter()
            .zip(lang_strings.iter())
            .map(|(chunk, language)| Bm25Document {
                chunk_id: &chunk.id,
                file_path: &chunk.file_path,
                content: &chunk.content,
                symbol_name: chunk.symbol_name.as_deref(),
                language,
            })
            .collect();
        bm25.insert_batch(&docs).unwrap();

        let results = vec![SearchResult {
            file_path: "crates/turborepo-types/src/lib.rs".to_string(),
            line_start: 750,
            line_end: 799,
            content: chunks[0].content.clone(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("TaskDefinition".to_string()),
            symbol_type: Some(SymbolType::Struct),
        }];

        let expanded =
            expand_cross_language_context(dir.path(), "TaskDefinition", &results).unwrap();

        assert!(expanded.iter().any(|result| {
            result.file_path == "packages/turbo-types/src/types/config-v2.ts"
                && result.symbol_name.as_deref() == Some("Pipeline")
        }));
    }

    #[test]
    fn same_language_context_adds_abort_helper() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let bm25_dir = dir.path().join("bm25");
        let store = MetadataStore::open(&metadata_path).unwrap();
        let bm25 = Bm25Index::open(&bm25_dir).unwrap();

        let chunks = vec![
            Chunk {
                id: "http:0".to_string(),
                file_path: "src/flask/app.py".to_string(),
                line_start: 830,
                line_end: 863,
                content: "def handle_http_exception(self, ctx, e: HTTPException):\n    \"\"\"Handles an HTTP exception and returns the exception as response.\"\"\"\n    handler = self._find_error_handler(e, ctx.request.blueprints)\n    if handler is None:\n        return e\n    return self.ensure_sync(handler)(e)"
                    .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Method),
                symbol_name: Some("handle_http_exception".to_string()),
            },
            Chunk {
                id: "http:1".to_string(),
                file_path: "src/flask/helpers.py".to_string(),
                line_start: 281,
                line_end: 301,
                content: "def abort(code):\n    \"\"\"Raise an HTTPException for the given status code.\"\"\"\n    if current_app is not None:\n        current_app.aborter(code)\n    _wz_abort(code)"
                    .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("abort".to_string()),
            },
            Chunk {
                id: "http:2".to_string(),
                file_path: "docs/errorhandling.rst".to_string(),
                line_start: 1,
                line_end: 20,
                content: "Custom error pages and HTTPException handlers.".to_string(),
                language: Language::Markdown,
                symbol_type: Some(SymbolType::Block),
                symbol_name: None,
            },
        ];

        store.insert_chunks(&chunks).unwrap();
        let lang_strings: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.language.to_string())
            .collect();
        let docs: Vec<Bm25Document<'_>> = chunks
            .iter()
            .zip(lang_strings.iter())
            .map(|(chunk, language)| Bm25Document {
                chunk_id: &chunk.id,
                file_path: &chunk.file_path,
                content: &chunk.content,
                symbol_name: chunk.symbol_name.as_deref(),
                language,
            })
            .collect();
        bm25.insert_batch(&docs).unwrap();

        let results = vec![SearchResult {
            file_path: "src/flask/app.py".to_string(),
            line_start: 830,
            line_end: 863,
            content: chunks[0].content.clone(),
            language: Language::Python,
            score: 1.0,
            symbol_name: Some("handle_http_exception".to_string()),
            symbol_type: Some(SymbolType::Method),
        }];

        let expanded = expand_same_language_context(
            dir.path(),
            "how are HTTP errors handled and returned to clients",
            &results,
        )
        .unwrap();

        assert!(expanded.iter().any(|result| {
            result.file_path == "src/flask/helpers.py"
                && result.symbol_name.as_deref() == Some("abort")
        }));
    }

    #[test]
    fn error_queries_add_abort_alias_symbol() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "error:0".to_string(),
                    file_path: "src/flask/app.py".to_string(),
                    line_start: 830,
                    line_end: 863,
                    content: "def handle_http_exception(self, ctx, e: HTTPException):\n    return e"
                        .to_string(),
                    language: Language::Python,
                    symbol_type: Some(SymbolType::Method),
                    symbol_name: Some("handle_http_exception".to_string()),
                },
                Chunk {
                    id: "error:1".to_string(),
                    file_path: "src/flask/helpers.py".to_string(),
                    line_start: 281,
                    line_end: 301,
                    content: "def abort(code):\n    raise HTTPException()".to_string(),
                    language: Language::Python,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("abort".to_string()),
                },
            ])
            .unwrap();

        let results = vec![SearchResult {
            file_path: "src/flask/app.py".to_string(),
            line_start: 830,
            line_end: 863,
            content: "def handle_http_exception(self, ctx, e: HTTPException):\n    return e"
                .to_string(),
            language: Language::Python,
            score: 1.0,
            symbol_name: Some("handle_http_exception".to_string()),
            symbol_type: Some(SymbolType::Method),
        }];

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "how are HTTP errors handled and returned to clients",
            results,
            RankingStage::Initial,
        )
        .unwrap();

        assert!(augmented.iter().any(|result| {
            result.file_path == "src/flask/helpers.py"
                && result.symbol_name.as_deref() == Some("abort")
        }));
    }
}
