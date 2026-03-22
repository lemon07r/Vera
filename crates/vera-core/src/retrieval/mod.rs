//! Hybrid retrieval pipeline: BM25 + vector search, RRF fusion, reranking.
//!
//! This module is responsible for:
//! - BM25 keyword search via Tantivy
//! - Vector similarity search via sqlite-vec
//! - Reciprocal Rank Fusion (RRF) for merging results
//! - Cross-encoder reranking via external API
//! - Post-retrieval filtering by language, path glob, and symbol type
//! - Graceful degradation when services are unavailable

pub mod bm25;
pub mod hybrid;
pub mod query_classifier;
pub mod reranker;
pub mod search_service;
pub mod vector;

pub use bm25::{search_bm25, search_bm25_with_stores};
pub use hybrid::{HybridSearchError, fuse_rrf, search_hybrid, search_hybrid_reranked};
pub use reranker::{
    ApiReranker, RerankScore, Reranker, RerankerConfig, RerankerError, rerank_results,
};

pub mod dynamic_reranker;
pub use dynamic_reranker::{DynamicReranker, create_dynamic_reranker};

pub mod local_reranker;

pub use local_reranker::LocalReranker;

pub use vector::{VectorSearchError, search_vector, search_vector_with_stores};

use crate::types::{SearchFilters, SearchResult};

#[cfg(test)]
#[path = "search_quality_tests.rs"]
mod search_quality_tests;

/// Apply search filters to a list of results, preserving order and limit.
///
/// Filters are applied post-retrieval: results that don't match all active
/// filters are removed. The `limit` parameter caps the final result count.
pub fn apply_filters(
    results: Vec<SearchResult>,
    filters: &SearchFilters,
    limit: usize,
) -> Vec<SearchResult> {
    if filters.is_empty() {
        let mut results = results;
        results.truncate(limit);
        return results;
    }

    results
        .into_iter()
        .filter(|r| filters.matches(r))
        .take(limit)
        .collect()
}
