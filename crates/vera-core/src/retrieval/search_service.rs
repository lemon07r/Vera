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

    if let Some(identifier_case) = extract_exact_identifier_case(query) {
        let mut chunks = store.get_chunks_by_symbol_name_case_sensitive(&identifier_case)?;
        let identifier = identifier_case.to_ascii_lowercase();
        let mut fallback_chunks = store.get_chunks_by_symbol_name(&identifier)?;
        fallback_chunks.retain(|chunk| {
            chunk.symbol_name.as_deref() != Some(identifier_case.as_str())
        });
        chunks.extend(fallback_chunks);
        chunks.sort_by(|a, b| {
            path_depth(&a.file_path)
                .cmp(&path_depth(&b.file_path))
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        supplemental.extend(chunks.into_iter().map(chunk_to_result));

        if wants_related_identifier_context(query) {
            let mut related = store.get_chunks_by_symbol_name_substring(&identifier_case, 256)?;
            related.retain(|chunk| {
                chunk.symbol_name
                    .as_deref()
                    .is_some_and(|name| !name.eq_ignore_ascii_case(&identifier_case))
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
    }

    supplemental.extend(expand_file_context(query, &store, &results)?);

    if supplemental.is_empty() {
        return Ok(results);
    }

    let mut merged = Vec::with_capacity(supplemental.len() + results.len());
    let mut seen = HashSet::new();

    for result in supplemental.into_iter().chain(results) {
        if seen.insert(result_key(&result)) {
            merged.push(result);
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
    let mut files = Vec::new();
    let mut seen = HashSet::new();

    for result in results.iter().take(6) {
        if !seen.insert(result.file_path.clone()) {
            continue;
        }
        if !is_expandable_file(result) {
            continue;
        }
        if result_needs_file_context(result) || file_matches_query_keywords(&result.file_path, &keywords)
        {
            files.push(result.file_path.clone());
        }
        if files.len() >= 3 {
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
        chunks.sort_by_key(structural_chunk_priority);
        supplemental.extend(chunks.into_iter().take(8).map(chunk_to_result));
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

fn related_symbol_priority(query: &str, chunk: &Chunk) -> (u8, u32) {
    let lower_query = query.to_ascii_lowercase();
    let lines = chunk.line_end.saturating_sub(chunk.line_start) + 1;
    let priority = if lower_query.contains("implement") {
        if chunk
            .symbol_name
            .as_deref()
            .is_some_and(|name| name.to_ascii_lowercase().contains("impl"))
        {
            0
        } else if matches!(chunk.symbol_type, Some(SymbolType::Struct | SymbolType::Class)) {
            1
        } else {
            2
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
        } else if matches!(chunk.symbol_type, Some(SymbolType::Class | SymbolType::Struct)) {
            1
        } else {
            2
        }
    } else if matches!(chunk.symbol_type, Some(SymbolType::Trait | SymbolType::Interface)) {
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

fn structural_chunk_priority(chunk: &Chunk) -> (u8, u32, u32) {
    let rank = match chunk.symbol_type {
        Some(SymbolType::Struct | SymbolType::Class | SymbolType::Trait | SymbolType::Interface) => 0,
        Some(SymbolType::Enum | SymbolType::Module) => 1,
        Some(SymbolType::Block) => 2,
        _ => 3,
    };
    let lines = chunk.line_end.saturating_sub(chunk.line_start) + 1;
    (rank, chunk.line_start, lines)
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

        let expanded =
            expand_file_context("request validation and schema enforcement", &store, &results)
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
}
