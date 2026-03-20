//! Hybrid retrieval pipeline: BM25 + vector search, RRF fusion, reranking.
//!
//! This module is responsible for:
//! - BM25 keyword search via Tantivy
//! - Vector similarity search via sqlite-vec
//! - Reciprocal Rank Fusion (RRF) for merging results
//! - Cross-encoder reranking via external API
//! - Graceful degradation when services are unavailable

pub mod bm25;
pub mod hybrid;
pub mod vector;

pub use bm25::{search_bm25, search_bm25_with_stores};
pub use hybrid::{HybridSearchError, fuse_rrf, search_hybrid};
pub use vector::{VectorSearchError, search_vector, search_vector_with_stores};
