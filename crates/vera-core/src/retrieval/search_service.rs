//! Shared search service used by both CLI and MCP.
//!
//! Encapsulates the common hybrid search flow: create embedding provider,
//! build reranker, compute fetch limits, execute search, apply filters.

use std::path::Path;

use anyhow::Result;
use tracing::warn;

use crate::config::VeraConfig;
use crate::retrieval::{apply_filters, search_bm25, search_hybrid, search_hybrid_reranked};
use crate::types::{SearchFilters, SearchResult};

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
    is_local: bool,
) -> Result<Vec<SearchResult>> {
    let fetch_limit = compute_fetch_limit(filters, result_limit);
    let rt = tokio::runtime::Runtime::new()?;

    // Try to create embedding provider for hybrid search.
    let (provider, model_name) =
        match rt.block_on(crate::embedding::create_dynamic_provider(config, is_local)) {
            Ok(res) => res,
            Err(e) => {
                warn!(
                    "Failed to create embedding provider ({}), using BM25-only search",
                    e
                );
                let results = search_bm25(index_dir, query, fetch_limit)?;
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
                stored_dim = dim;
            }
        }
    }

    let provider = crate::embedding::CachedEmbeddingProvider::new(provider, 512);

    // Create optional reranker.
    let reranker = rt
        .block_on(crate::retrieval::create_dynamic_reranker(config, is_local))
        .unwrap_or_else(|e| {
            warn!("Failed to create reranker ({})", e);
            None
        });

    let rrf_k = config.retrieval.rrf_k;
    let rerank_candidates = config.retrieval.rerank_candidates;

    let results = if let Some(ref reranker) = reranker {
        rt.block_on(search_hybrid_reranked(
            index_dir,
            &provider,
            reranker,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
            rerank_candidates.max(fetch_limit),
        ))?
    } else {
        rt.block_on(search_hybrid(
            index_dir,
            &provider,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
        ))?
    };

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
