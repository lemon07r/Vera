//! Tree-sitter query search over indexed files.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

use crate::corpus::{ContentClass, classify_content};
use crate::parsing::languages;
use crate::retrieval::file_scan::{allows_class, language_for_path};
use crate::storage::metadata::MetadataStore;
use crate::types::{Chunk, Language, SearchFilters, SearchResult, SymbolType};

#[derive(Debug, Clone, Copy)]
struct MatchSpan {
    start_byte: usize,
    end_byte: usize,
    start_line: u32,
    end_line: u32,
}

/// Search indexed files with a raw tree-sitter query.
pub fn search_ast_query(
    index_dir: &Path,
    query_text: &str,
    language: Language,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    if limit == 0 {
        bail!("limit must be greater than zero");
    }

    let grammar = languages::tree_sitter_grammar(language).with_context(|| {
        format!(
            "language '{}' does not support tree-sitter queries",
            language
        )
    })?;
    let query = Query::new(&grammar, query_text).map_err(|err| {
        anyhow::anyhow!(
            "invalid tree-sitter query at row {}, column {}: {}",
            err.row + 1,
            err.column + 1,
            err.message
        )
    })?;

    let metadata_path = index_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path)?;
    let mut files = store.indexed_files()?;
    files.sort();

    let repo_root = index_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine project root from index dir"))?;

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .context("failed to load tree-sitter grammar")?;

    let mut cursor = QueryCursor::new();
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for file_rel in files {
        if results.len() >= limit {
            break;
        }

        let file_language = language_for_path(&file_rel);
        if file_language != language || !filters.matches_file(&file_rel, file_language) {
            continue;
        }

        let file_abs = repo_root.join(&file_rel);
        let content = match std::fs::read_to_string(&file_abs) {
            Ok(content) => content,
            Err(_) => continue,
        };

        let class = classify_content(&file_rel, file_language, &content);
        if !allows_class(filters, class) {
            continue;
        }
        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        let Some(tree) = parser.parse(&content, None) else {
            continue;
        };

        let file_chunks = if filters.symbol_type.is_some() {
            Some(store.get_chunks_by_file(&file_rel)?)
        } else {
            None
        };

        let mut query_matches = cursor.matches(&query, tree.root_node(), content.as_bytes());
        loop {
            query_matches.advance();
            let Some(query_match) = query_matches.get() else {
                break;
            };
            if results.len() >= limit {
                break;
            }

            let Some(span) = match_span(query_match.captures) else {
                continue;
            };
            let line_start = span.start_line;
            let line_end = span.end_line;
            let key = format!("{file_rel}:{line_start}:{line_end}");
            if !seen.insert(key) {
                continue;
            }

            let snippet = content
                .get(span.start_byte..span.end_byte)
                .map(str::to_string)
                .unwrap_or_default();
            if snippet.trim().is_empty() {
                continue;
            }

            let (symbol_name, symbol_type) =
                symbol_for_range(file_chunks.as_deref(), line_start, line_end);
            if !filters.matches_symbol_type(symbol_type) {
                continue;
            }

            results.push(SearchResult {
                file_path: file_rel.clone(),
                line_start,
                line_end,
                content: snippet,
                language,
                score: 1.0,
                symbol_name,
                symbol_type,
            });
        }
    }

    Ok(results)
}

fn match_span(captures: &[tree_sitter::QueryCapture<'_>]) -> Option<MatchSpan> {
    let mut start_byte = None;
    let mut end_byte = None;
    let mut start_line = None;
    let mut end_line = None;

    for capture in captures {
        let node = capture.node;
        let node_start = node.start_position();
        let node_end = node.end_position();
        start_byte = Some(match start_byte {
            Some(current) if current <= node.start_byte() => current,
            _ => node.start_byte(),
        });
        end_byte = Some(match end_byte {
            Some(current) if current >= node.end_byte() => current,
            _ => node.end_byte(),
        });
        start_line = Some(match start_line {
            Some(current) if current <= node_start.row as u32 + 1 => current,
            _ => node_start.row as u32 + 1,
        });
        end_line = Some(match end_line {
            Some(current) if current > node_end.row as u32 => current,
            _ => node_end.row as u32 + 1,
        });
    }

    Some(MatchSpan {
        start_byte: start_byte?,
        end_byte: end_byte?,
        start_line: start_line?,
        end_line: end_line?,
    })
}

fn symbol_for_range(
    chunks: Option<&[Chunk]>,
    line_start: u32,
    line_end: u32,
) -> (Option<String>, Option<SymbolType>) {
    chunks
        .and_then(|chunks| {
            chunks
                .iter()
                .filter(|chunk| {
                    chunk.line_start <= line_start
                        && line_end <= chunk.line_end
                        && (chunk.symbol_type.is_some() || chunk.symbol_name.is_some())
                })
                .min_by_key(|chunk| {
                    (
                        chunk.line_end.saturating_sub(chunk.line_start),
                        chunk.line_start,
                        chunk.line_end,
                    )
                })
                .map(|chunk| (chunk.symbol_name.clone(), chunk.symbol_type))
        })
        .unwrap_or((None, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VeraConfig;
    use crate::embedding::test_helpers::MockProvider;
    use crate::indexing::index_repository;

    #[tokio::test]
    async fn ast_query_finds_rust_function_items() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("main.rs"),
            "fn alpha() {}\nfn beta() { println!(\"hi\"); }\n",
        )
        .unwrap();

        let provider = MockProvider::new(8);
        let config = VeraConfig::default();
        index_repository(dir.path(), &provider, &config, "mock-model")
            .await
            .unwrap();

        let index_dir = crate::indexing::index_dir(dir.path());
        let results = search_ast_query(
            &index_dir,
            "(function_item name: (identifier) @fn)",
            Language::Rust,
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .any(|result| result.content.contains("alpha"))
        );
        assert!(results.iter().any(|result| result.content.contains("beta")));
    }

    #[tokio::test]
    async fn ast_query_skips_generated_files_by_default() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src").join("main.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(
            dir.path().join("src").join("generated.rs"),
            "fn generated_fn() {}\n",
        )
        .unwrap();

        let provider = MockProvider::new(8);
        let config = VeraConfig::default();
        index_repository(dir.path(), &provider, &config, "mock-model")
            .await
            .unwrap();

        let index_dir = crate::indexing::index_dir(dir.path());
        let results = search_ast_query(
            &index_dir,
            "(function_item name: (identifier) @fn)",
            Language::Rust,
            10,
            &SearchFilters {
                include_generated: Some(false),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "src/main.rs");
    }
}
