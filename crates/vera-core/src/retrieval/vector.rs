//! Vector similarity search over indexed chunks.
//!
//! Provides semantic search using the embedding pipeline and sqlite-vec
//! vector store. Generates a query embedding via the configured API provider,
//! then performs nearest-neighbor lookup, and hydrates results from the
//! metadata store. Finds semantically related code even when query terms
//! don't appear literally in results (e.g., "memory allocation" finds `alloc`).

use std::path::Path;

use anyhow::Result;
use tracing::debug;

use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;
use crate::types::SearchResult;

/// Errors specific to vector search.
#[derive(Debug, thiserror::Error)]
pub enum VectorSearchError {
    /// The embedding API is unavailable (connection, auth, etc.).
    #[error("embedding API unavailable: {source}")]
    EmbeddingUnavailable {
        #[from]
        source: EmbeddingError,
    },

    /// Storage or metadata error.
    #[error("storage error: {0}")]
    StorageError(#[from] anyhow::Error),
}

/// Perform a vector similarity search over the indexed chunks.
///
/// Opens the vector store and metadata store from the index directory,
/// generates a query embedding via the provider, performs nearest-neighbor
/// search, and returns hydrated results sorted by similarity (descending).
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `provider` — Embedding provider for generating the query vector
/// - `query` — The search query text
/// - `limit` — Maximum number of results to return
/// - `stored_dim` — Dimensionality of stored vectors (for truncation matching)
///
/// # Returns
/// A vector of `SearchResult` with full chunk metadata, sorted by similarity
/// score descending.
pub async fn search_vector(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    query: &str,
    limit: usize,
    stored_dim: usize,
) -> Result<Vec<SearchResult>, VectorSearchError> {
    let vector_path = index_dir.join("vectors.db");
    let metadata_path = index_dir.join("metadata.db");

    let vector_store = VectorStore::open(&vector_path, stored_dim)
        .map_err(|e| VectorSearchError::StorageError(e.context("failed to open vector store")))?;
    let metadata_store = MetadataStore::open(&metadata_path)
        .map_err(|e| VectorSearchError::StorageError(e.context("failed to open metadata store")))?;

    search_vector_with_stores(&vector_store, &metadata_store, provider, query, limit).await
}

/// Perform vector search using pre-opened stores (useful for testing and reuse).
///
/// Generates a query embedding, runs nearest-neighbor search in the vector
/// store, then hydrates each result with full chunk metadata. Results are
/// returned sorted by similarity score in descending order.
///
/// The similarity score is derived from the distance: `score = 1.0 / (1.0 + distance)`.
/// This transforms the raw distance (lower = closer) into a 0–1 similarity score
/// (higher = more similar), which is consistent with other scoring conventions
/// in the retrieval pipeline.
pub async fn search_vector_with_stores(
    vector_store: &VectorStore,
    metadata_store: &MetadataStore,
    provider: &impl EmbeddingProvider,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, VectorSearchError> {
    // 1. Generate query embedding.
    let query_embedding = generate_query_embedding(provider, query, vector_store.dim()).await?;

    debug!(
        query = query,
        dim = query_embedding.len(),
        "generated query embedding"
    );

    // 2. Search the vector store for nearest neighbors.
    // Fetch more candidates than the limit to account for missing metadata.
    let candidates = limit.saturating_mul(2).max(limit + 10);

    let vector_results = vector_store
        .search(&query_embedding, candidates)
        .map_err(|e| VectorSearchError::StorageError(e.context("vector search failed")))?;

    debug!(
        query = query,
        raw_results = vector_results.len(),
        "vector search returned candidates"
    );

    // 3. Hydrate results from metadata store and convert distances to scores.
    let mut results = Vec::with_capacity(vector_results.len());

    for vr in &vector_results {
        let chunk = metadata_store.get_chunk(&vr.chunk_id).map_err(|e| {
            VectorSearchError::StorageError(e.context(format!(
                "failed to fetch metadata for chunk: {}",
                vr.chunk_id
            )))
        })?;

        let Some(chunk) = chunk else {
            debug!(
                chunk_id = %vr.chunk_id,
                "chunk metadata not found, skipping"
            );
            continue;
        };

        // Convert distance to similarity score: higher is better.
        let score = distance_to_similarity(vr.distance);

        results.push(SearchResult {
            file_path: chunk.file_path,
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            content: chunk.content,
            language: chunk.language,
            score,
            symbol_name: chunk.symbol_name,
            symbol_type: chunk.symbol_type,
        });

        if results.len() >= limit {
            break;
        }
    }

    debug!(
        query = query,
        returned = results.len(),
        "vector search complete"
    );

    Ok(results)
}

/// Generate a query embedding, truncating to match stored dimensionality.
///
/// The embedding provider may return vectors of higher dimensionality than
/// what is stored (e.g., Qwen3 produces 4096-dim but we store 1024-dim
/// via Matryoshka truncation). This function truncates the query vector
/// to match the stored vector dimensionality.
async fn generate_query_embedding(
    provider: &impl EmbeddingProvider,
    query: &str,
    stored_dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let texts = vec![query.to_string()];
    let mut vectors = provider.embed_batch(&texts).await?;

    if vectors.is_empty() {
        return Err(EmbeddingError::ResponseError {
            message: "embedding API returned no vectors for query".to_string(),
        });
    }

    let mut vector = vectors.swap_remove(0);

    // Truncate to stored dimensionality if needed (Matryoshka embedding).
    if vector.len() > stored_dim {
        vector.truncate(stored_dim);
    }

    // Validate dimensionality matches.
    if vector.len() != stored_dim {
        return Err(EmbeddingError::ResponseError {
            message: format!(
                "query embedding dimension {} does not match stored dimension {}",
                vector.len(),
                stored_dim
            ),
        });
    }

    Ok(vector)
}

/// Convert a distance score to a similarity score.
///
/// sqlite-vec returns L2 (Euclidean) distances where lower is better.
/// We convert to a similarity score in (0, 1] where higher is better:
///   similarity = 1.0 / (1.0 + distance)
///
/// This is a standard transformation that:
/// - Maps distance=0 to similarity=1 (perfect match)
/// - Maps distance→∞ to similarity→0 (no similarity)
/// - Is monotonically decreasing (preserves ranking)
fn distance_to_similarity(distance: f64) -> f64 {
    1.0 / (1.0 + distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::test_helpers::MockProvider;
    use crate::types::{Chunk, Language, SymbolType};

    /// Create sample chunks with semantic variety for testing.
    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                id: "src/alloc.rs:0".to_string(),
                file_path: "src/alloc.rs".to_string(),
                line_start: 1,
                line_end: 15,
                content: "pub fn alloc_page(size: usize) -> *mut u8 {\n    \
                           let layout = Layout::from_size_align(size, 4096).unwrap();\n    \
                           unsafe { std::alloc::alloc(layout) }\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("alloc_page".to_string()),
            },
            Chunk {
                id: "src/alloc.rs:1".to_string(),
                file_path: "src/alloc.rs".to_string(),
                line_start: 17,
                line_end: 30,
                content: "pub fn dealloc_page(ptr: *mut u8, size: usize) {\n    \
                           let layout = Layout::from_size_align(size, 4096).unwrap();\n    \
                           unsafe { std::alloc::dealloc(ptr, layout) }\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("dealloc_page".to_string()),
            },
            Chunk {
                id: "src/auth.rs:0".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 1,
                line_end: 12,
                content: "pub fn authenticate(user: &str, password: &str) -> Result<Token> {\n    \
                           let hash = hash_password(password);\n    \
                           verify_credentials(user, &hash)\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("authenticate".to_string()),
            },
            Chunk {
                id: "src/db.py:0".to_string(),
                file_path: "src/db.py".to_string(),
                line_start: 1,
                line_end: 10,
                content: "class DatabaseConnection:\n    \
                           def __init__(self, host, port):\n        \
                           self.conn = psycopg2.connect(host=host, port=port)\n    \
                           def execute_query(self, sql):\n        \
                           return self.conn.execute(sql)"
                    .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Class),
                symbol_name: Some("DatabaseConnection".to_string()),
            },
            Chunk {
                id: "src/cache.go:0".to_string(),
                file_path: "src/cache.go".to_string(),
                line_start: 1,
                line_end: 15,
                content: "func NewLRUCache(capacity int) *LRUCache {\n    \
                           return &LRUCache{\n        \
                           capacity: capacity,\n        \
                           items: make(map[string]*list.Element),\n        \
                           order: list.New(),\n    }\n}"
                    .to_string(),
                language: Language::Go,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("NewLRUCache".to_string()),
            },
            Chunk {
                id: "src/server.ts:0".to_string(),
                file_path: "src/server.ts".to_string(),
                line_start: 1,
                line_end: 10,
                content: "function handleRequest(req: Request): Response {\n    \
                           const auth = authenticate(req.headers);\n    \
                           if (!auth) return new Response('Unauthorized', { status: 401 });\n    \
                           return processRequest(req);\n}"
                    .to_string(),
                language: Language::TypeScript,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("handleRequest".to_string()),
            },
        ]
    }

    /// Embed sample chunks with MockProvider and store in vector + metadata stores.
    async fn setup_test_stores(dim: usize) -> (VectorStore, MetadataStore) {
        let chunks = sample_chunks();
        let provider = MockProvider::new(dim);

        // Store metadata.
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        metadata_store.insert_chunks(&chunks).unwrap();

        // Generate embeddings and store vectors.
        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        let embeddings = crate::embedding::embed_chunks(&provider, &chunks, chunks.len())
            .await
            .unwrap();

        let batch: Vec<(&str, &[f32])> = embeddings
            .iter()
            .map(|(id, vec)| (id.as_str(), vec.as_slice()))
            .collect();
        vector_store.insert_batch(&batch).unwrap();

        (vector_store, metadata_store)
    }

    // ── distance_to_similarity tests ─────────────────────────────────

    #[test]
    fn distance_zero_gives_similarity_one() {
        let score = distance_to_similarity(0.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_positive_gives_score_between_zero_and_one() {
        let score = distance_to_similarity(1.0);
        assert!(score > 0.0);
        assert!(score < 1.0);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_large_gives_score_near_zero() {
        let score = distance_to_similarity(1000.0);
        assert!(score > 0.0);
        assert!(score < 0.01);
    }

    #[test]
    fn similarity_is_monotonically_decreasing() {
        let distances = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
        let scores: Vec<f64> = distances
            .iter()
            .map(|d| distance_to_similarity(*d))
            .collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] > scores[i],
                "similarity must decrease as distance increases: {} > {} at index {i}",
                scores[i - 1],
                scores[i],
            );
        }
    }

    // ── generate_query_embedding tests ───────────────────────────────

    #[tokio::test]
    async fn query_embedding_has_correct_dimension() {
        let provider = MockProvider::new(8);
        let embedding = generate_query_embedding(&provider, "test query", 8)
            .await
            .unwrap();
        assert_eq!(embedding.len(), 8);
    }

    #[tokio::test]
    async fn query_embedding_truncates_to_stored_dim() {
        // Provider produces 16-dim, but stored is 8-dim.
        let provider = MockProvider::new(16);
        let embedding = generate_query_embedding(&provider, "test query", 8)
            .await
            .unwrap();
        assert_eq!(embedding.len(), 8);
    }

    #[tokio::test]
    async fn query_embedding_rejects_dim_mismatch() {
        // Provider produces 4-dim, but stored is 8-dim.
        let provider = MockProvider::new(4);
        let result = generate_query_embedding(&provider, "test query", 8).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("dimension"),
            "error should mention dimension: {err}"
        );
    }

    #[tokio::test]
    async fn query_embedding_propagates_auth_error() {
        let provider = MockProvider::failing(EmbeddingError::AuthError {
            message: "invalid key".to_string(),
        });
        let result = generate_query_embedding(&provider, "test query", 4).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EmbeddingError::AuthError { .. }),
            "should propagate auth error"
        );
    }

    #[tokio::test]
    async fn query_embedding_propagates_connection_error() {
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "unreachable".to_string(),
        });
        let result = generate_query_embedding(&provider, "test query", 4).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EmbeddingError::ConnectionError { .. }),
            "should propagate connection error"
        );
    }

    // ── search_vector_with_stores tests ──────────────────────────────

    #[tokio::test]
    async fn search_returns_results_for_indexed_content() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "alloc", 10)
                .await
                .unwrap();

        assert!(!results.is_empty(), "should find results for 'alloc'");
    }

    #[tokio::test]
    async fn results_sorted_by_score_descending() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "database connection query",
            10,
        )
        .await
        .unwrap();

        assert!(
            results.len() >= 2,
            "need multiple results to verify ordering"
        );
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores must be descending: {} >= {} at position {i}",
                results[i - 1].score,
                results[i].score,
            );
        }
    }

    #[tokio::test]
    async fn results_include_full_metadata() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "cache", 10)
                .await
                .unwrap();

        assert!(!results.is_empty());
        let top = &results[0];
        // Every result should have required fields populated.
        assert!(!top.file_path.is_empty(), "file_path should be set");
        assert!(top.line_start > 0, "line_start should be 1-based");
        assert!(top.line_end >= top.line_start, "line_end >= line_start");
        assert!(!top.content.is_empty(), "content should be present");
        assert!(top.score > 0.0, "score should be positive");
        assert!(top.score <= 1.0, "similarity score should be <= 1.0");
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "function", 2)
                .await
                .unwrap();

        assert!(results.len() <= 2, "results should respect the limit of 2");
    }

    #[tokio::test]
    async fn scores_are_positive_and_bounded() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "authenticate",
            10,
        )
        .await
        .unwrap();

        for result in &results {
            assert!(result.score > 0.0, "score should be positive");
            assert!(result.score <= 1.0, "score should be <= 1.0");
        }
    }

    #[tokio::test]
    async fn search_with_embedding_error_returns_error() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "API down".to_string(),
        });

        let result =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "test", 10).await;

        assert!(result.is_err(), "should return error when embedding fails");
        assert!(
            matches!(
                result.unwrap_err(),
                VectorSearchError::EmbeddingUnavailable { .. }
            ),
            "should be an embedding unavailable error"
        );
    }

    #[tokio::test]
    async fn search_returns_results_from_multiple_languages() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        // Get all results.
        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "code function",
            10,
        )
        .await
        .unwrap();

        let languages: std::collections::HashSet<_> = results.iter().map(|r| r.language).collect();
        assert!(
            languages.len() >= 2,
            "should return results from multiple languages, got: {languages:?}"
        );
    }

    #[tokio::test]
    async fn search_with_truncation() {
        // Provider returns 16-dim vectors, but store uses 8-dim.
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        // Use a provider that produces larger vectors than stored.
        let query_provider = MockProvider::new(16);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &query_provider, "cache", 10)
                .await
                .unwrap();

        assert!(
            !results.is_empty(),
            "should work with truncated query embeddings"
        );
    }

    #[tokio::test]
    async fn empty_vector_store_returns_empty_results() {
        let dim = 8;
        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "anything", 10)
                .await
                .unwrap();

        assert!(results.is_empty(), "empty store should return no results");
    }

    #[tokio::test]
    async fn result_content_matches_source_chunk() {
        let dim = 8;
        let chunks = sample_chunks();
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "alloc page", 10)
                .await
                .unwrap();

        assert!(!results.is_empty());
        // Find a result from alloc.rs and verify content matches original.
        let alloc_result = results.iter().find(|r| r.file_path == "src/alloc.rs");
        assert!(alloc_result.is_some(), "should find result from alloc.rs");
        let alloc_result = alloc_result.unwrap();
        // Verify content matches one of the original chunks.
        let matching_chunk = chunks
            .iter()
            .find(|c| c.file_path == alloc_result.file_path && c.content == alloc_result.content);
        assert!(
            matching_chunk.is_some(),
            "result content should match a source chunk"
        );
    }
}
