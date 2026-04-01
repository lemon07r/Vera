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
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::embedding::EmbeddingProvider;
use crate::retrieval::bm25::search_bm25;
use crate::retrieval::query_classifier::{QueryType, classify_query};
use crate::retrieval::ranking::is_path_weighted_query;
use crate::retrieval::reranker::{Reranker, rerank_results};
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

/// Compute the number of vector candidates to fetch for a given limit and
/// query type multiplier. Ensures at least 50 candidates for any limit.
pub fn compute_vector_candidates(limit: usize, multiplier: usize) -> usize {
    limit.saturating_mul(multiplier).max(50)
}

fn compute_bm25_candidates(query: &str, limit: usize) -> usize {
    let query_type = classify_query(query);
    let token_count = query.split_whitespace().count();

    if is_path_weighted_query(query) {
        return limit.saturating_mul(5).max(100);
    }
    if query_type == QueryType::NaturalLanguage {
        return limit.saturating_mul(4).max(limit + 40);
    }
    if token_count <= 2 {
        return limit.saturating_mul(4).max(80);
    }

    limit.saturating_mul(3).max(limit + 20)
}

/// Per-stage timing data from hybrid search.
#[derive(Debug, Default)]
pub struct HybridTimings {
    pub embedding: Option<Duration>,
    pub bm25: Option<Duration>,
    pub vector: Option<Duration>,
    pub fusion: Option<Duration>,
    pub reranking: Option<Duration>,
}

/// Perform hybrid search combining BM25 and vector retrieval via RRF fusion.
///
/// Runs BM25 and vector search, merges results using Reciprocal Rank Fusion,
/// and returns the top results. If vector search fails (e.g., embedding API
/// unavailable), falls back to BM25-only results with a warning.
pub async fn search_hybrid(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    query: &str,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let bm25_candidates = compute_bm25_candidates(query, limit);
    let mut timings = HybridTimings::default();

    let bm25_start = Instant::now();
    let bm25_results = search_bm25(index_dir, query, bm25_candidates);
    timings.bm25 = Some(bm25_start.elapsed());

    let embed_start = Instant::now();
    let vector_results =
        search_vector(index_dir, provider, query, vector_candidates, stored_dim).await;
    let vector_elapsed = embed_start.elapsed();
    timings.embedding = Some(vector_elapsed);
    timings.vector = Some(vector_elapsed);

    match (bm25_results, vector_results) {
        (Ok(bm25), Ok(vector)) => {
            debug!(
                bm25_count = bm25.len(),
                vector_count = vector.len(),
                "merging BM25 and vector results via RRF"
            );
            let fusion_start = Instant::now();
            let fused = fuse_rrf(&bm25, &vector, rrf_k, limit);
            timings.fusion = Some(fusion_start.elapsed());
            Ok((fused, timings))
        }
        (Ok(bm25), Err(vec_err)) => {
            warn!(
                error = %vec_err,
                "vector search failed, falling back to BM25-only results"
            );
            let mut results = bm25;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Ok(vector)) => {
            warn!(
                error = %bm25_err,
                "BM25 search failed, falling back to vector-only results"
            );
            let mut results = vector;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Err(vec_err)) => Err(HybridSearchError::BothFailed {
            bm25_error: format!("{bm25_err:#}"),
            vector_error: format!("{vec_err:#}"),
        }),
    }
}

/// Perform hybrid search with cross-encoder reranking.
///
/// Runs the full hybrid pipeline (BM25 + vector → RRF fusion), then
/// sends the top candidates to a cross-encoder reranker for more accurate
/// relevance scoring.
///
/// **Graceful degradation:**
/// - If the reranker API is unavailable (timeout, 5xx, connection error),
///   returns unreranked results with a warning logged to stderr.
/// - If the embedding API is unavailable, falls back to BM25-only results
///   (handled by the inner `search_hybrid` call).
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `provider` — Embedding provider for vector search
/// - `reranker` — Reranker for result refinement
/// - `query` — The search query text
/// - `limit` — Maximum number of results to return
/// - `rrf_k` — RRF constant (typically 60.0)
/// - `stored_dim` — Dimensionality of stored vectors
/// - `rerank_candidates` — Number of candidates to send to the reranker
/// - `vector_candidates` — Number of vector candidates to fetch (query-type-aware)
#[allow(clippy::too_many_arguments)]
pub async fn search_hybrid_reranked(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    query: &str,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let fetch_limit = rerank_candidates.max(limit);

    let (hybrid_results, mut timings) = search_hybrid(
        index_dir,
        provider,
        query,
        fetch_limit,
        rrf_k,
        stored_dim,
        vector_candidates,
    )
    .await?;

    if hybrid_results.is_empty() {
        return Ok((hybrid_results, timings));
    }

    let rerank_start = Instant::now();
    match rerank_results(reranker, query, &hybrid_results, rerank_candidates).await {
        Ok(mut reranked) => {
            timings.reranking = Some(rerank_start.elapsed());
            info!(
                query = query,
                candidates = hybrid_results.len(),
                reranked = reranked.len(),
                "reranking complete"
            );
            reranked.truncate(limit);
            Ok((reranked, timings))
        }
        Err(rerank_err) => {
            timings.reranking = Some(rerank_start.elapsed());
            warn!(
                error = %rerank_err,
                "reranker unavailable, returning unreranked results"
            );
            eprintln!(
                "Warning: reranker unavailable ({rerank_err}), returning unreranked results."
            );
            let mut results = hybrid_results;
            results.truncate(limit);
            Ok((results, timings))
        }
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
    fuse_rrf_multi(&[bm25_results, vector_results], rrf_k, limit)
}

/// Fuse multiple ranked result lists with reciprocal rank fusion (RRF).
pub fn fuse_rrf_multi(
    result_sets: &[&[SearchResult]],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut fused: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for result_set in result_sets {
        for (rank_0, result) in result_set.iter().enumerate() {
            let key = result_key(result);
            let rrf_score = 1.0 / (rrf_k + (rank_0 + 1) as f64);

            fused
                .entry(key)
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert_with(|| (rrf_score, result.clone()));
        }
    }

    let mut ranked: Vec<(f64, SearchResult)> = fused.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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

    // ── Integration tests for search_hybrid_reranked ─────────────────

    /// Set up a temp index directory with BM25 + vector + metadata stores.
    /// Returns (index_dir_path, stored_dim) for use in integration tests.
    async fn setup_test_index(tmp: &std::path::Path) -> (std::path::PathBuf, usize) {
        use crate::config::VeraConfig;
        use crate::embedding::embed_chunks;
        use crate::embedding::test_helpers::MockProvider;
        use crate::parsing;
        use crate::storage::bm25::{Bm25Document, Bm25Index};
        use crate::storage::metadata::MetadataStore;
        use crate::storage::vector::VectorStore;

        let dim = 8;
        let provider = MockProvider::new(dim);
        let config = VeraConfig::default();

        // Create sample source files.
        let repo_dir = tmp.join("repo");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(
            repo_dir.join("auth.rs"),
            "pub fn authenticate(user: &str, pass: &str) -> Result<Token, Error> {\n    \
             let hash = hash_password(pass);\n    verify_credentials(user, &hash)\n}\n\n\
             pub fn authorize(token: &Token, resource: &str) -> bool {\n    \
             token.has_permission(resource)\n}\n",
        )
        .unwrap();
        std::fs::write(
            repo_dir.join("cache.rs"),
            "pub fn get_cached(key: &str) -> Option<Value> {\n    \
             CACHE.lock().unwrap().get(key).cloned()\n}\n\n\
             pub fn set_cached(key: &str, value: Value) {\n    \
             CACHE.lock().unwrap().insert(key.to_string(), value);\n}\n",
        )
        .unwrap();

        // Parse and chunk.
        let mut all_chunks = Vec::new();
        for file in ["auth.rs", "cache.rs"] {
            let source = std::fs::read_to_string(repo_dir.join(file)).unwrap();
            let lang = crate::types::Language::Rust;
            let chunks = parsing::parse_and_chunk(&source, file, lang, &config.indexing).unwrap();
            all_chunks.extend(chunks);
        }

        // Create index directory and stores.
        let index_dir = repo_dir.join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Metadata store.
        let metadata_path = index_dir.join("metadata.db");
        let metadata_store = MetadataStore::open(&metadata_path).unwrap();
        metadata_store.insert_chunks(&all_chunks).unwrap();

        // Vector store.
        let vector_path = index_dir.join("vectors.db");
        let vector_store = VectorStore::open(&vector_path, dim).unwrap();
        let embeddings = embed_chunks(&provider, &all_chunks, all_chunks.len(), 0)
            .await
            .unwrap();
        let batch: Vec<(&str, &[f32])> = embeddings
            .iter()
            .map(|(id, vec)| (id.as_str(), vec.as_slice()))
            .collect();
        vector_store.insert_batch(&batch).unwrap();

        // BM25 index.
        let bm25_dir = index_dir.join("bm25");
        let bm25 = Bm25Index::open(&bm25_dir).unwrap();
        let lang_strings: Vec<String> = all_chunks.iter().map(|c| c.language.to_string()).collect();
        let bm25_docs: Vec<Bm25Document<'_>> = all_chunks
            .iter()
            .zip(lang_strings.iter())
            .map(|(c, lang)| Bm25Document {
                chunk_id: &c.id,
                file_path: &c.file_path,
                content: &c.content,
                symbol_name: c.symbol_name.as_deref(),
                language: lang,
            })
            .collect();
        bm25.insert_batch(&bm25_docs).unwrap();

        (index_dir, dim)
    }

    #[tokio::test]
    async fn search_hybrid_reranked_returns_reranked_results() {
        use crate::embedding::test_helpers::MockProvider;
        use crate::retrieval::reranker::test_helpers::MockReranker;

        let tmp = tempfile::tempdir().unwrap();
        let (index_dir, dim) = setup_test_index(tmp.path()).await;

        let provider = MockProvider::new(dim);
        let reranker = MockReranker::new();

        let (results, _timings) = search_hybrid_reranked(
            &index_dir,
            &provider,
            &reranker,
            "authenticate",
            5,
            60.0,
            dim,
            10,
            50,
        )
        .await
        .unwrap();

        assert!(
            !results.is_empty(),
            "should find results for 'authenticate'"
        );

        // Results should be sorted by reranker scores (descending).
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "reranked scores must be descending: {} >= {}",
                results[i - 1].score,
                results[i].score,
            );
        }
    }

    #[tokio::test]
    async fn search_hybrid_reranked_degrades_on_reranker_failure() {
        use crate::embedding::test_helpers::MockProvider;
        use crate::retrieval::reranker::RerankerError;
        use crate::retrieval::reranker::test_helpers::MockReranker;

        let tmp = tempfile::tempdir().unwrap();
        let (index_dir, dim) = setup_test_index(tmp.path()).await;

        let provider = MockProvider::new(dim);
        let reranker = MockReranker::failing(RerankerError::ConnectionError {
            message: "reranker timeout".to_string(),
        });

        // Should NOT return an error — graceful degradation returns unreranked results.
        let (results, _timings) = search_hybrid_reranked(
            &index_dir,
            &provider,
            &reranker,
            "authenticate",
            5,
            60.0,
            dim,
            10,
            50,
        )
        .await
        .unwrap();

        assert!(
            !results.is_empty(),
            "should return unreranked results when reranker fails"
        );
    }

    #[tokio::test]
    async fn search_hybrid_reranked_degrades_on_embedding_failure() {
        use crate::embedding::EmbeddingError;
        use crate::embedding::test_helpers::MockProvider;
        use crate::retrieval::reranker::test_helpers::MockReranker;

        let tmp = tempfile::tempdir().unwrap();
        let (index_dir, dim) = setup_test_index(tmp.path()).await;

        // Embedding provider that always fails.
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "embedding API down".to_string(),
        });
        let reranker = MockReranker::new();

        // Should fall back to BM25-only results (not crash/hang).
        let (results, _timings) = search_hybrid_reranked(
            &index_dir,
            &provider,
            &reranker,
            "authenticate",
            5,
            60.0,
            dim,
            10,
            50,
        )
        .await
        .unwrap();

        // BM25 fallback should still find keyword matches.
        assert!(
            !results.is_empty(),
            "should return BM25-only results when embedding API fails"
        );
    }

    #[tokio::test]
    async fn search_hybrid_reranked_respects_limit() {
        use crate::embedding::test_helpers::MockProvider;
        use crate::retrieval::reranker::test_helpers::MockReranker;

        let tmp = tempfile::tempdir().unwrap();
        let (index_dir, dim) = setup_test_index(tmp.path()).await;

        let provider = MockProvider::new(dim);
        let reranker = MockReranker::new();

        let (results, _timings) = search_hybrid_reranked(
            &index_dir, &provider, &reranker, "function", 2, 60.0, dim, 10, 50,
        )
        .await
        .unwrap();

        assert!(results.len() <= 2, "should respect the limit of 2");
    }

    // ── compute_vector_candidates tests ─────────────────────────────

    #[test]
    fn compute_vector_candidates_minimum_50() {
        // Even with small limit and multiplier, floor is 50.
        assert!(compute_vector_candidates(5, 3) >= 50);
        assert!(compute_vector_candidates(1, 1) >= 50);
        assert!(compute_vector_candidates(10, 3) >= 50);
    }

    #[test]
    fn compute_vector_candidates_default_limit_10() {
        // For default limit=10 with identifier multiplier (3): max(30, 50) = 50
        assert_eq!(compute_vector_candidates(10, 3), 50);
        // For default limit=10 with NL multiplier (5): max(50, 50) = 50
        assert_eq!(compute_vector_candidates(10, 5), 50);
    }

    #[test]
    fn compute_vector_candidates_scales_with_limit() {
        // For large limit, multiplier should dominate.
        assert_eq!(compute_vector_candidates(100, 3), 300);
        assert_eq!(compute_vector_candidates(100, 5), 500);
    }

    #[test]
    fn path_queries_expand_bm25_candidate_pool() {
        assert_eq!(
            compute_bm25_candidates("turbo.json pipeline configuration", 20),
            100
        );
    }

    #[test]
    fn natural_language_queries_expand_bm25_candidate_pool() {
        assert_eq!(
            compute_bm25_candidates("request validation and schema enforcement", 20),
            80
        );
    }

    #[test]
    fn short_identifier_queries_expand_bm25_candidate_pool() {
        assert_eq!(compute_bm25_candidates("Config", 20), 80);
    }

    // ── Integration: NL vs identifier query produces different fusion ──

    #[tokio::test]
    async fn nl_query_uses_different_rrf_k_than_identifier() {
        use crate::embedding::test_helpers::MockProvider;
        use crate::retrieval::query_classifier::{
            QueryType, classify_query, params_for_query_type,
        };
        use crate::retrieval::reranker::test_helpers::MockReranker;

        let tmp = tempfile::tempdir().unwrap();
        let (index_dir, dim) = setup_test_index(tmp.path()).await;

        let provider = MockProvider::new(dim);
        let reranker = MockReranker::new();

        // Identifier query → uses default k=60.
        let id_query = "authenticate";
        let id_type = classify_query(id_query);
        assert_eq!(id_type, QueryType::Identifier);
        let id_params = params_for_query_type(id_type);

        let (id_results, _) = search_hybrid_reranked(
            &index_dir,
            &provider,
            &reranker,
            id_query,
            5,
            id_params.rrf_k,
            dim,
            10,
            compute_vector_candidates(5, id_params.vector_candidate_multiplier),
        )
        .await
        .unwrap();

        // NL query → uses lower k=20.
        let nl_query = "how is authentication handled";
        let nl_type = classify_query(nl_query);
        assert_eq!(nl_type, QueryType::NaturalLanguage);
        let nl_params = params_for_query_type(nl_type);

        let (nl_results, _) = search_hybrid_reranked(
            &index_dir,
            &provider,
            &reranker,
            nl_query,
            5,
            nl_params.rrf_k,
            dim,
            10,
            compute_vector_candidates(5, nl_params.vector_candidate_multiplier),
        )
        .await
        .unwrap();

        // Both should return results (the index has auth content).
        assert!(
            !id_results.is_empty(),
            "identifier query should find results"
        );
        assert!(!nl_results.is_empty(), "NL query should find results");

        // The key assertion: different RRF k was used.
        assert!(
            id_params.rrf_k > nl_params.rrf_k,
            "identifier k ({}) should be greater than NL k ({})",
            id_params.rrf_k,
            nl_params.rrf_k
        );
    }
}
