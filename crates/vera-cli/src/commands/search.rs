//! `vera search <query>` — Search the indexed codebase.

use std::path::Path;
use std::process;

use crate::helpers::output_results;

/// Run the `vera search <query>` command.
///
/// Performs hybrid search (BM25 + vector via RRF fusion) with optional
/// cross-encoder reranking. Falls back gracefully:
/// - Embedding API unavailable → BM25-only search with warning
/// - Reranker API unavailable → unreranked hybrid results with warning
pub fn run(
    query: &str,
    limit: Option<usize>,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
) -> anyhow::Result<()> {
    let config = vera_core::config::VeraConfig::default();
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);

    // Find the index directory (look in current working directory).
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    let index_dir = vera_core::indexing::index_dir(&cwd);

    if !index_dir.exists() {
        eprintln!(
            "Error: no index found in current directory.\n\
             Hint: run `vera index <path>` first to create an index."
        );
        process::exit(1);
    }

    // Build the tokio runtime for async embedding/reranker calls.
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    // Create the embedding provider from environment.
    let provider_config = match vera_core::embedding::EmbeddingProviderConfig::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!(
                "Warning: embedding API not configured ({err}), falling back to BM25-only search."
            );
            return run_bm25_fallback(&index_dir, query, result_limit, filters, json_output);
        }
    };
    let provider_config = provider_config
        .with_timeout(std::time::Duration::from_secs(
            config.embedding.timeout_secs,
        ))
        .with_max_retries(config.embedding.max_retries);

    let provider = match vera_core::embedding::OpenAiProvider::new(provider_config) {
        Ok(p) => p,
        Err(err) => {
            eprintln!(
                "Warning: failed to initialize embedding provider ({err}), \
                 falling back to BM25-only search."
            );
            return run_bm25_fallback(&index_dir, query, result_limit, filters, json_output);
        }
    };

    // Wrap with query embedding cache for fast repeated queries.
    let provider = vera_core::embedding::CachedEmbeddingProvider::new(provider, 512);

    // Create the reranker from environment (optional).
    let reranker = create_reranker(&config);

    // Fetch more candidates when filters are active.
    let fetch_limit = if filters.is_empty() {
        result_limit
    } else {
        result_limit.saturating_mul(3).max(result_limit + 20)
    };

    // Run hybrid search with optional reranking.
    let stored_dim = config.embedding.max_stored_dim;
    let rrf_k = config.retrieval.rrf_k;
    let rerank_candidates = config.retrieval.rerank_candidates;

    let results = if let Some(ref reranker) = reranker {
        match rt.block_on(vera_core::retrieval::search_hybrid_reranked(
            &index_dir,
            &provider,
            reranker,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
            rerank_candidates.max(fetch_limit),
        )) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("Error: search failed: {err:#}");
                process::exit(1);
            }
        }
    } else {
        match rt.block_on(vera_core::retrieval::search_hybrid(
            &index_dir,
            &provider,
            query,
            fetch_limit,
            rrf_k,
            stored_dim,
        )) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("Error: search failed: {err:#}");
                process::exit(1);
            }
        }
    };

    // Apply post-retrieval filters.
    let results = vera_core::retrieval::apply_filters(results, filters, result_limit);

    output_results(&results, json_output);
    Ok(())
}

/// Create the optional reranker from environment.
fn create_reranker(
    config: &vera_core::config::VeraConfig,
) -> Option<vera_core::retrieval::ApiReranker> {
    if !config.retrieval.reranking_enabled {
        return None;
    }

    match vera_core::retrieval::RerankerConfig::from_env() {
        Ok(reranker_config) => {
            let reranker_config = reranker_config
                .with_timeout(std::time::Duration::from_secs(30))
                .with_max_retries(2);
            match vera_core::retrieval::ApiReranker::new(reranker_config) {
                Ok(r) => Some(r),
                Err(err) => {
                    eprintln!(
                        "Warning: failed to initialize reranker ({err}), \
                         search will proceed without reranking."
                    );
                    None
                }
            }
        }
        Err(err) => {
            tracing::debug!(error = %err, "reranker not configured, skipping reranking");
            None
        }
    }
}

/// BM25-only fallback when embedding API is unavailable.
fn run_bm25_fallback(
    index_dir: &Path,
    query: &str,
    limit: usize,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
) -> anyhow::Result<()> {
    let fetch_limit = if filters.is_empty() {
        limit
    } else {
        limit.saturating_mul(3).max(limit + 20)
    };

    let results = match vera_core::retrieval::search_bm25(index_dir, query, fetch_limit) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("Error: BM25 search failed: {err:#}");
            process::exit(1);
        }
    };

    let results = vera_core::retrieval::apply_filters(results, filters, limit);
    output_results(&results, json_output);
    Ok(())
}
