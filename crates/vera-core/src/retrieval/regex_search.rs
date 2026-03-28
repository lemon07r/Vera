//! Regex pattern search over indexed files.
//!
//! Searches file contents using a regex pattern, returning matches with
//! file paths, line numbers, and surrounding context. Operates on the
//! indexed file list from the metadata store, respecting the same file
//! scope as semantic search.

use std::path::Path;

use anyhow::Result;
use regex::RegexBuilder;

use crate::types::{Language, SearchResult};

/// Search indexed files for a regex pattern.
///
/// Reads the file list from the metadata store, then greps each file.
/// Returns results as `SearchResult` for compatibility with the rest of
/// the pipeline.
pub fn search_regex(
    index_dir: &Path,
    pattern: &str,
    limit: usize,
    case_insensitive: bool,
    context_lines: usize,
) -> Result<Vec<SearchResult>> {
    let regex = RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid regex pattern: {e}"))?;

    let metadata_path = index_dir.join("metadata.db");
    let store = crate::storage::metadata::MetadataStore::open(&metadata_path)?;
    let files = store.indexed_files()?;

    // Resolve the project root (index_dir is .vera/, parent is project root).
    let project_root = index_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine project root from index dir"))?;

    let mut results = Vec::new();

    for file_rel in &files {
        if results.len() >= limit {
            break;
        }

        let file_abs = project_root.join(file_rel);
        let content = match std::fs::read_to_string(&file_abs) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if results.len() >= limit {
                break;
            }
            if !regex.is_match(line) {
                continue;
            }

            let ctx_start = i.saturating_sub(context_lines);
            let ctx_end = (i + context_lines + 1).min(lines.len());
            let snippet: String = lines[ctx_start..ctx_end].join("\n");

            let lang = std::path::Path::new(file_rel)
                .extension()
                .and_then(|e| e.to_str())
                .map(Language::from_extension)
                .unwrap_or(Language::Unknown);

            results.push(SearchResult {
                file_path: file_rel.clone(),
                line_start: (ctx_start + 1) as u32,
                line_end: ctx_end as u32,
                content: snippet,
                score: 1.0,
                symbol_name: None,
                symbol_type: None,
                language: lang,
            });
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_regex_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let index_dir = tmp.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let result = search_regex(&index_dir, "[invalid", 10, false, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid regex"));
    }
}
