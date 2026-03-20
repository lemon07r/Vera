//! Hybrid retrieval pipeline: BM25 + vector search, RRF fusion, reranking.
//!
//! This module is responsible for:
//! - BM25 keyword search via Tantivy
//! - Vector similarity search via sqlite-vec
//! - Reciprocal Rank Fusion (RRF) for merging results
//! - Cross-encoder reranking via external API
//! - Graceful degradation when services are unavailable

pub mod bm25;

pub use bm25::{search_bm25, search_bm25_with_stores};

#[cfg(test)]
mod tests {
    #[test]
    fn module_loads() {
        // Placeholder: will be replaced with real retrieval tests.
    }
}
