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
use crate::retrieval::hybrid::compute_vector_candidates;
use crate::retrieval::query_classifier::{classify_query, params_for_query_type};
use crate::retrieval::ranking::{RankingStage, apply_query_ranking, is_path_weighted_query};
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
    let total_start = Instant::now();
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
                let bm25_start = Instant::now();
                let results = apply_query_ranking(
                    query,
                    search_bm25(index_dir, query, fetch_limit)?,
                    RankingStage::Initial,
                );
                let timings = SearchTimings {
                    bm25: Some(bm25_start.elapsed()),
                    total: Some(total_start.elapsed()),
                    ..Default::default()
                };
                return Ok((apply_filters(results, filters, result_limit), timings));
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
            if !crate::config::model_names_match(&s_model, &model_name) {
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

    let ranking_stage = if reranker.is_some() {
        RankingStage::PostRerank
    } else {
        RankingStage::Initial
    };

    let (results, hybrid_timings) = if let Some(ref reranker) = reranker {
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

    let mut timings = SearchTimings {
        embedding: hybrid_timings.embedding,
        bm25: hybrid_timings.bm25,
        vector: hybrid_timings.vector,
        fusion: hybrid_timings.fusion,
        reranking: hybrid_timings.reranking,
        ..Default::default()
    };

    let aug_start = Instant::now();
    let results = augment_exact_match_candidates(index_dir, query, results, ranking_stage)?;
    timings.augmentation = Some(aug_start.elapsed());

    timings.total = Some(total_start.elapsed());
    Ok((apply_filters(results, filters, result_limit), timings))
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
    _query: &str,
) -> usize {
    compute_vector_candidates(fetch_limit, query_params.vector_candidate_multiplier)
}

fn effective_rerank_candidates(base: usize, fetch_limit: usize, _query: &str) -> usize {
    base.max(fetch_limit)
}

fn augment_exact_match_candidates(
    index_dir: &Path,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
) -> Result<Vec<SearchResult>> {
    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = crate::storage::metadata::MetadataStore::open(&metadata_path) else {
        return Ok(apply_query_ranking(query, results, stage));
    };

    let mut supplemental = Vec::new();

    // Direct filename lookup for path-weighted queries (e.g. "Cargo.toml workspace config").
    if let Some(filename) = extract_exact_filename(query).filter(|_| is_path_weighted_query(query))
    {
        let mut matching_files: Vec<String> = store
            .indexed_files()?
            .into_iter()
            .filter(|path| file_name(path).eq_ignore_ascii_case(&filename))
            .collect();
        matching_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));

        for file_path in matching_files.into_iter().take(20) {
            supplemental.extend(
                store
                    .get_chunks_by_file(&file_path)?
                    .into_iter()
                    .map(chunk_to_result),
            );
        }
    }

    // Direct symbol lookup for identifier queries (e.g. "Config", "Blueprint class").
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
        chunks.sort_by(|a, b| {
            exact_match_priority(query, identifier_case, a)
                .cmp(&exact_match_priority(query, identifier_case, b))
                .then(path_depth(&a.file_path).cmp(&path_depth(&b.file_path)))
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        supplemental.extend(chunks.into_iter().map(chunk_to_result));
    }

    if supplemental.is_empty() {
        return Ok(apply_query_ranking(query, results, stage));
    }

    // Merge: supplemental first (exact matches), then original results, deduped.
    let mut merged = Vec::with_capacity(supplemental.len() + results.len());
    let mut seen = HashSet::new();

    for result in supplemental.into_iter().chain(results) {
        if seen.insert(result_key(&result)) {
            merged.push(result);
        }
    }

    Ok(apply_query_ranking(query, merged, stage))
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
    fn effective_candidates_use_base_multipliers() {
        // Rerank candidates just return base.max(fetch_limit)
        assert_eq!(effective_rerank_candidates(50, 10, "anything"), 50);
        assert_eq!(effective_rerank_candidates(5, 10, "anything"), 10);

        // Vector candidates use query_params multiplier without inflation
        let nl_params =
            params_for_query_type(crate::retrieval::query_classifier::QueryType::NaturalLanguage);
        let vc = effective_vector_candidates(10, nl_params, "some query");
        assert!(vc >= 50); // at least the minimum from compute_vector_candidates
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

        let augmented =
            augment_exact_match_candidates(dir.path(), "Config", Vec::new(), RankingStage::Initial)
                .unwrap();

        assert_eq!(
            augmented[0].file_path,
            "crates/searcher/src/searcher/mod.rs"
        );
    }
}
