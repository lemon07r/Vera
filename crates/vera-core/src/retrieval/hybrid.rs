//! Hybrid retrieval engine combining BM25 and vector search via RRF fusion.
//!
//! Runs both BM25 keyword search and vector similarity search in parallel,
//! then merges the results using Reciprocal Rank Fusion (RRF). Items appearing
//! in both result sets rank higher than single-source results.
//!
//! RRF score: `score = sum(1 / (k + rank_i))` where `k` is a constant
//! (typically 60) and `rank_i` is the 1-based rank of the item in each
//! source result list.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use tracing::{debug, warn};

use crate::embedding::EmbeddingProvider;
use crate::retrieval::bm25::search_bm25;
use crate::retrieval::vector::search_vector;
use crate::types::SearchResult;

/// Errors specific to hybrid search.
#[derive(Debug, thiserror::Error)]
pub enum HybridSearchError {
    /// Both BM25 and vector search failed.
    #[error("both BM25 and vector search failed: bm25={bm25_error}, vector={vector_error}")]
    BothFailed {
        bm25_error: String,
        vector_error: String,
    },

    /// Storage or pipeline error.
    #[error("{0}")]
    PipelineError(#[from] anyhow::Error),
}

/// Perform hybrid search combining BM25 and vector retrieval via RRF fusion.
///
/// Runs BM25 and vector search, merges results using Reciprocal Rank Fusion,
/// and returns the top results. If vector search fails (e.g., embedding API
/// unavailable), falls back to BM25-only results with a warning.
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `provider` — Embedding provider for vector search query embedding
/// - `query` — The search query text
/// - `limit` — Maximum number of results to return
/// - `rrf_k` — RRF constant (typically 60.0)
/// - `stored_dim` — Dimensionality of stored vectors
pub async fn search_hybrid(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    query: &str,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
) -> Result<Vec<SearchResult>, HybridSearchError> {
    // Fetch more candidates from each source for better fusion quality.
    let candidates = limit.saturating_mul(3).max(limit + 20);

    // Run BM25 search.
    let bm25_results = search_bm25(index_dir, query, candidates);

    // Run vector search.
    let vector_results = search_vector(index_dir, provider, query, candidates, stored_dim).await;

    // Handle failure modes with graceful degradation.
    match (bm25_results, vector_results) {
        (Ok(bm25), Ok(vector)) => {
            debug!(
                bm25_count = bm25.len(),
                vector_count = vector.len(),
                "merging BM25 and vector results via RRF"
            );
            Ok(fuse_rrf(&bm25, &vector, rrf_k, limit))
        }
        (Ok(bm25), Err(vec_err)) => {
            warn!(
                error = %vec_err,
                "vector search failed, falling back to BM25-only results"
            );
            let mut results = bm25;
            results.truncate(limit);
            Ok(results)
        }
        (Err(bm25_err), Ok(vector)) => {
            warn!(
                error = %bm25_err,
                "BM25 search failed, falling back to vector-only results"
            );
            let mut results = vector;
            results.truncate(limit);
            Ok(results)
        }
        (Err(bm25_err), Err(vec_err)) => Err(HybridSearchError::BothFailed {
            bm25_error: format!("{bm25_err:#}"),
            vector_error: format!("{vec_err:#}"),
        }),
    }
}

/// Perform hybrid search using pre-opened stores (useful for testing).
///
/// Takes pre-computed BM25 and vector results and fuses them via RRF.
/// This is the core fusion logic, separated from I/O for testability.
pub fn fuse_rrf(
    bm25_results: &[SearchResult],
    vector_results: &[SearchResult],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    // Build a map of chunk_key → (rrf_score, SearchResult).
    // We key by (file_path, line_start, line_end) since chunk IDs aren't
    // in SearchResult but these fields uniquely identify a chunk.
    let mut fused: HashMap<String, (f64, SearchResult)> = HashMap::new();

    // Process BM25 results (1-based ranking).
    for (rank_0, result) in bm25_results.iter().enumerate() {
        let key = result_key(result);
        let rrf_score = 1.0 / (rrf_k + (rank_0 + 1) as f64);

        fused
            .entry(key)
            .and_modify(|(score, _)| *score += rrf_score)
            .or_insert_with(|| (rrf_score, result.clone()));
    }

    // Process vector results (1-based ranking).
    for (rank_0, result) in vector_results.iter().enumerate() {
        let key = result_key(result);
        let rrf_score = 1.0 / (rrf_k + (rank_0 + 1) as f64);

        fused
            .entry(key)
            .and_modify(|(score, _)| *score += rrf_score)
            .or_insert_with(|| (rrf_score, result.clone()));
    }

    // Sort by RRF score descending.
    let mut ranked: Vec<(f64, SearchResult)> = fused.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top results, replacing original scores with RRF scores.
    ranked
        .into_iter()
        .take(limit)
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result
        })
        .collect()
}

/// Generate a unique key for a search result to detect overlaps.
///
/// Uses file_path + line_start + line_end as a composite key, since
/// SearchResult doesn't carry the chunk ID but these fields uniquely
/// identify a chunk within the index.
fn result_key(result: &SearchResult) -> String {
    format!(
        "{}:{}:{}",
        result.file_path, result.line_start, result.line_end
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, SymbolType};

    /// Helper to create a SearchResult with given parameters.
    fn make_result(
        file: &str,
        line_start: u32,
        line_end: u32,
        score: f64,
        symbol_name: Option<&str>,
    ) -> SearchResult {
        SearchResult {
            file_path: file.to_string(),
            line_start,
            line_end,
            content: format!("content of {file}:{line_start}"),
            language: Language::Rust,
            score,
            symbol_name: symbol_name.map(|s| s.to_string()),
            symbol_type: Some(SymbolType::Function),
        }
    }

    // ── RRF score calculation tests ─────────────────────────────────

    #[test]
    fn rrf_single_source_bm25_only() {
        let bm25 = vec![
            make_result("a.rs", 1, 10, 5.0, Some("func_a")),
            make_result("b.rs", 1, 10, 3.0, Some("func_b")),
        ];
        let vector: Vec<SearchResult> = vec![];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        assert_eq!(results.len(), 2);
        // First BM25 result: 1/(60+1) ≈ 0.01639
        let expected_score_1 = 1.0 / 61.0;
        assert!(
            (results[0].score - expected_score_1).abs() < 1e-10,
            "first result RRF score: got {}, expected {}",
            results[0].score,
            expected_score_1
        );
        assert_eq!(results[0].file_path, "a.rs");
    }

    #[test]
    fn rrf_single_source_vector_only() {
        let bm25: Vec<SearchResult> = vec![];
        let vector = vec![
            make_result("c.rs", 1, 10, 0.9, Some("func_c")),
            make_result("d.rs", 1, 10, 0.7, Some("func_d")),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        assert_eq!(results.len(), 2);
        let expected_score_1 = 1.0 / 61.0;
        assert!(
            (results[0].score - expected_score_1).abs() < 1e-10,
            "first result RRF score: got {}, expected {}",
            results[0].score,
            expected_score_1
        );
        assert_eq!(results[0].file_path, "c.rs");
    }

    #[test]
    fn rrf_overlapping_results_rank_higher() {
        // Result "shared.rs:1:10" appears in both BM25 (rank 2) and vector (rank 1).
        // Result "bm25_only.rs:1:10" appears only in BM25 (rank 1).
        // Result "vector_only.rs:1:10" appears only in vector (rank 2).
        let bm25 = vec![
            make_result("bm25_only.rs", 1, 10, 5.0, Some("bm25_func")),
            make_result("shared.rs", 1, 10, 3.0, Some("shared_func")),
        ];
        let vector = vec![
            make_result("shared.rs", 1, 10, 0.9, Some("shared_func")),
            make_result("vector_only.rs", 1, 10, 0.7, Some("vector_func")),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        // shared.rs appears in both: RRF = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        let shared_score = 1.0 / 62.0 + 1.0 / 61.0;
        // bm25_only.rs: RRF = 1/(60+1) = 1/61
        let bm25_only_score = 1.0 / 61.0;
        // vector_only.rs: RRF = 1/(60+2) = 1/62
        let _vector_only_score = 1.0 / 62.0;

        assert!(
            shared_score > bm25_only_score,
            "overlapping result should have higher RRF score"
        );

        // shared.rs should be the top result.
        assert_eq!(
            results[0].file_path, "shared.rs",
            "result appearing in both lists should rank first"
        );
        assert!(
            (results[0].score - shared_score).abs() < 1e-10,
            "shared score: got {}, expected {}",
            results[0].score,
            shared_score
        );
    }

    #[test]
    fn rrf_scores_are_descending() {
        let bm25 = vec![
            make_result("a.rs", 1, 10, 5.0, Some("func_a")),
            make_result("b.rs", 1, 10, 3.0, Some("func_b")),
            make_result("c.rs", 1, 10, 1.0, Some("func_c")),
        ];
        let vector = vec![
            make_result("d.rs", 1, 10, 0.9, Some("func_d")),
            make_result("a.rs", 1, 10, 0.8, Some("func_a")),
            make_result("e.rs", 1, 10, 0.5, Some("func_e")),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores must be descending: {} >= {} at position {i}",
                results[i - 1].score,
                results[i].score,
            );
        }
    }

    #[test]
    fn rrf_respects_limit() {
        let bm25 = vec![
            make_result("a.rs", 1, 10, 5.0, None),
            make_result("b.rs", 1, 10, 3.0, None),
            make_result("c.rs", 1, 10, 1.0, None),
        ];
        let vector = vec![
            make_result("d.rs", 1, 10, 0.9, None),
            make_result("e.rs", 1, 10, 0.7, None),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 2);
        assert_eq!(results.len(), 2, "should respect the limit");
    }

    #[test]
    fn rrf_empty_inputs_return_empty() {
        let results = fuse_rrf(&[], &[], 60.0, 10);
        assert!(results.is_empty(), "no inputs should give no results");
    }

    #[test]
    fn rrf_preserves_metadata_from_first_seen() {
        let bm25 = vec![make_result("shared.rs", 1, 10, 5.0, Some("func"))];
        let vector = vec![make_result("shared.rs", 1, 10, 0.9, Some("func"))];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.file_path, "shared.rs");
        assert_eq!(result.line_start, 1);
        assert_eq!(result.line_end, 10);
        assert_eq!(result.symbol_name.as_deref(), Some("func"));
        assert_eq!(result.symbol_type, Some(SymbolType::Function));
        assert_eq!(result.language, Language::Rust);
    }

    #[test]
    fn rrf_k_parameter_affects_scores() {
        let bm25 = vec![make_result("a.rs", 1, 10, 5.0, None)];
        let vector: Vec<SearchResult> = vec![];

        // With k=60: score = 1/61 ≈ 0.01639
        let results_k60 = fuse_rrf(&bm25, &vector, 60.0, 10);
        // With k=1: score = 1/2 = 0.5
        let results_k1 = fuse_rrf(&bm25, &vector, 1.0, 10);

        assert!(
            results_k1[0].score > results_k60[0].score,
            "lower k should produce higher scores: k1={}, k60={}",
            results_k1[0].score,
            results_k60[0].score
        );
    }

    #[test]
    fn rrf_with_known_inputs_produces_exact_scores() {
        // RRF with k=60:
        // Item A: BM25 rank 1, vector rank 3 → 1/61 + 1/63
        // Item B: BM25 rank 2, vector rank 1 → 1/62 + 1/61
        // Item C: BM25 rank 3, vector rank 2 → 1/63 + 1/62
        let bm25 = vec![
            make_result("a.rs", 1, 10, 5.0, Some("a")),
            make_result("b.rs", 1, 10, 3.0, Some("b")),
            make_result("c.rs", 1, 10, 1.0, Some("c")),
        ];
        let vector = vec![
            make_result("b.rs", 1, 10, 0.9, Some("b")),
            make_result("c.rs", 1, 10, 0.8, Some("c")),
            make_result("a.rs", 1, 10, 0.5, Some("a")),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        let score_a = 1.0 / 61.0 + 1.0 / 63.0;
        let score_b = 1.0 / 62.0 + 1.0 / 61.0;
        let score_c = 1.0 / 63.0 + 1.0 / 62.0;

        // B should rank first (highest combined score)
        assert_eq!(results[0].file_path, "b.rs");
        assert!(
            (results[0].score - score_b).abs() < 1e-10,
            "B score: got {}, expected {score_b}",
            results[0].score
        );

        // A should rank second
        assert_eq!(results[1].file_path, "a.rs");
        assert!(
            (results[1].score - score_a).abs() < 1e-10,
            "A score: got {}, expected {score_a}",
            results[1].score
        );

        // C should rank third
        assert_eq!(results[2].file_path, "c.rs");
        assert!(
            (results[2].score - score_c).abs() < 1e-10,
            "C score: got {}, expected {score_c}",
            results[2].score
        );
    }

    #[test]
    fn rrf_distinct_results_no_overlap() {
        // When BM25 and vector have completely different results,
        // BM25 rank-1 and vector rank-1 should tie (same RRF score).
        let bm25 = vec![
            make_result("bm25_1.rs", 1, 10, 5.0, None),
            make_result("bm25_2.rs", 1, 10, 3.0, None),
        ];
        let vector = vec![
            make_result("vec_1.rs", 1, 10, 0.9, None),
            make_result("vec_2.rs", 1, 10, 0.7, None),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        assert_eq!(results.len(), 4);
        // The top two should have score 1/61, the bottom two should have score 1/62.
        let expected_top = 1.0 / 61.0;
        let expected_bottom = 1.0 / 62.0;

        assert!((results[0].score - expected_top).abs() < 1e-10);
        assert!((results[1].score - expected_top).abs() < 1e-10);
        assert!((results[2].score - expected_bottom).abs() < 1e-10);
        assert!((results[3].score - expected_bottom).abs() < 1e-10);
    }

    #[test]
    fn rrf_different_chunks_same_file_are_separate() {
        // Two different chunks from the same file should be treated as separate.
        let bm25 = vec![
            make_result("lib.rs", 1, 10, 5.0, Some("func_1")),
            make_result("lib.rs", 20, 30, 3.0, Some("func_2")),
        ];
        let vector = vec![make_result("lib.rs", 1, 10, 0.9, Some("func_1"))];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        // lib.rs:1:10 appears in both → higher score
        // lib.rs:20:30 appears only in BM25 → lower score
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].line_start, 1,
            "overlapping chunk should rank first"
        );
        assert_eq!(
            results[1].line_start, 20,
            "single-source chunk should rank second"
        );
    }

    #[test]
    fn rrf_scores_are_positive() {
        let bm25 = vec![
            make_result("a.rs", 1, 10, 5.0, None),
            make_result("b.rs", 1, 10, 3.0, None),
        ];
        let vector = vec![
            make_result("c.rs", 1, 10, 0.9, None),
            make_result("a.rs", 1, 10, 0.8, None),
        ];

        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        for result in &results {
            assert!(result.score > 0.0, "RRF scores should be positive");
        }
    }

    // ── result_key tests ────────────────────────────────────────────

    #[test]
    fn result_key_is_unique_for_different_chunks() {
        let r1 = make_result("a.rs", 1, 10, 1.0, None);
        let r2 = make_result("a.rs", 20, 30, 1.0, None);
        let r3 = make_result("b.rs", 1, 10, 1.0, None);

        assert_ne!(result_key(&r1), result_key(&r2));
        assert_ne!(result_key(&r1), result_key(&r3));
        assert_ne!(result_key(&r2), result_key(&r3));
    }

    #[test]
    fn result_key_is_same_for_same_chunk() {
        let r1 = make_result("a.rs", 1, 10, 5.0, Some("func"));
        let r2 = make_result("a.rs", 1, 10, 0.9, Some("func"));

        assert_eq!(result_key(&r1), result_key(&r2));
    }
}
