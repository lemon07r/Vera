//! Iterative (multi-hop) search: runs an initial semantic search, extracts
//! symbol names from the top results, and performs follow-up searches to
//! find related code. Merges and deduplicates all results.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::config::{InferenceBackend, VeraConfig};
use crate::types::{SearchFilters, SearchResult};

use super::search_service::{SearchTimings, execute_search};

/// Run an iterative (multi-hop) search.
///
/// 1. Execute the initial query via `execute_search`.
/// 2. Extract unique symbol names from the top results.
/// 3. Run follow-up searches for each extracted symbol.
/// 4. Merge and deduplicate, preserving the original result order first.
pub fn execute_iterative_search(
    index_dir: &Path,
    query: &str,
    config: &VeraConfig,
    filters: &SearchFilters,
    result_limit: usize,
    backend: InferenceBackend,
    hops: usize,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let fetch_per_hop = result_limit;

    let (initial_results, timings) =
        execute_search(index_dir, query, config, filters, fetch_per_hop, backend)?;

    if hops == 0 || initial_results.is_empty() {
        return Ok((initial_results, timings));
    }

    let mut seen = HashSet::new();
    let mut merged: Vec<SearchResult> = Vec::new();

    for r in &initial_results {
        let key = result_key(r);
        if seen.insert(key) {
            merged.push(r.clone());
        }
    }

    // Extract symbol names from initial results for follow-up queries.
    let follow_up_symbols: Vec<String> = initial_results
        .iter()
        .filter_map(|r| r.symbol_name.clone())
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            // Skip generic names that would produce noisy results.
            !matches!(
                lower.as_str(),
                "main" | "new" | "default" | "test" | "init" | "run" | "setup"
            )
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .take(5)
        .collect();

    for symbol in &follow_up_symbols {
        let (hop_results, _) = execute_search(
            index_dir,
            symbol,
            config,
            filters,
            fetch_per_hop / 2,
            backend,
        )?;

        for r in hop_results {
            let key = result_key(&r);
            if seen.insert(key) {
                merged.push(r);
            }
        }
    }

    merged.truncate(result_limit);
    Ok((merged, timings))
}

fn result_key(r: &SearchResult) -> String {
    format!("{}:{}:{}", r.file_path, r.line_start, r.line_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn result_key_format() {
        let r = SearchResult {
            file_path: "src/main.rs".to_string(),
            line_start: 10,
            line_end: 20,
            content: String::new(),
            score: 1.0,
            symbol_name: None,
            symbol_type: None,
            language: Language::Rust,
        };
        assert_eq!(result_key(&r), "src/main.rs:10:20");
    }
}
