//! Regex pattern search over indexed files.
//!
//! Searches file contents using a regex pattern, returning matches with
//! file paths, line numbers, and surrounding context. Operates on the
//! indexed file list from the metadata store, respecting the same file
//! scope as semantic search.

use std::path::Path;

use anyhow::Result;
use regex::{Match, RegexBuilder};

use crate::corpus::{ContentClass, classify_content, classify_path, is_minified_content};
use crate::retrieval::ranking::{RankingStage, apply_query_ranking_with_filters};
use crate::types::{Language, SearchFilters, SearchResult};

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
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let regex = RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid regex pattern: {e}"))?;

    let metadata_path = index_dir.join("metadata.db");
    let store = crate::storage::metadata::MetadataStore::open(&metadata_path)?;
    let mut files = store.indexed_files()?;
    files.sort_by(|left, right| {
        let left_key = {
            let language = language_for_path(left);
            let class = classify_path(left, language);
            (file_scan_priority(class, filters), path_depth(left))
        };
        let right_key = {
            let language = language_for_path(right);
            let class = classify_path(right, language);
            (file_scan_priority(class, filters), path_depth(right))
        };
        left_key.cmp(&right_key).then_with(|| left.cmp(right))
    });

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
        let language = language_for_path(file_rel);
        let class = classify_content(file_rel, language, &content);

        if !allows_class(filters, class) {
            continue;
        }

        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        if is_minified_content(&content) || matches!(class, ContentClass::Generated) {
            collect_minified_matches(&mut results, &regex, file_rel, &content, language, limit);
            continue;
        }

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
            let snippet = lines[ctx_start..ctx_end].join("\n");

            results.push(SearchResult {
                file_path: file_rel.clone(),
                line_start: (ctx_start + 1) as u32,
                line_end: ctx_end as u32,
                content: snippet,
                score: 1.0,
                symbol_name: None,
                symbol_type: None,
                language,
            });
        }
    }

    Ok(apply_query_ranking_with_filters(
        pattern,
        results,
        RankingStage::Initial,
        filters,
    ))
}

fn collect_minified_matches(
    results: &mut Vec<SearchResult>,
    regex: &regex::Regex,
    file_rel: &str,
    content: &str,
    language: Language,
    limit: usize,
) {
    for found in regex.find_iter(content) {
        if results.len() >= limit {
            break;
        }
        let (snippet, line_start, line_end) = bounded_match_snippet(content, found, 220);
        results.push(SearchResult {
            file_path: file_rel.to_string(),
            line_start,
            line_end,
            content: snippet,
            score: 1.0,
            symbol_name: None,
            symbol_type: None,
            language,
        });
    }
}

fn bounded_match_snippet(content: &str, found: Match<'_>, window: usize) -> (String, u32, u32) {
    let start = clamp_char_boundary(content, found.start().saturating_sub(window));
    let end = clamp_char_boundary(content, (found.end() + window).min(content.len()));
    let mut snippet = content[start..end].to_string();

    if start > 0 {
        snippet.insert_str(0, "...");
    }
    if end < content.len() {
        snippet.push_str("...");
    }

    let line_start = byte_to_line(content, found.start());
    let line_end = byte_to_line(content, found.end());
    (snippet, line_start, line_end.max(line_start))
}

fn clamp_char_boundary(content: &str, mut idx: usize) -> usize {
    while idx > 0 && !content.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn byte_to_line(content: &str, byte_idx: usize) -> u32 {
    content[..byte_idx.min(content.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count() as u32
        + 1
}

fn allows_class(filters: &SearchFilters, class: ContentClass) -> bool {
    match filters.scope {
        Some(scope) => {
            crate::corpus::matches_scope(class, scope, filters.include_generated.unwrap_or(true))
        }
        None => true,
    }
}

fn file_scan_priority(class: ContentClass, filters: &SearchFilters) -> u8 {
    match filters.scope {
        Some(crate::types::SearchScope::Docs) => match class {
            ContentClass::Docs => 0,
            ContentClass::Archive => 1,
            ContentClass::Config => 2,
            ContentClass::Source | ContentClass::Unknown => 3,
            ContentClass::Test | ContentClass::Example | ContentClass::Bench => 4,
            ContentClass::Runtime => 5,
            ContentClass::Generated => 6,
        },
        Some(crate::types::SearchScope::Runtime) => match class {
            ContentClass::Runtime => 0,
            ContentClass::Generated => 1,
            ContentClass::Source | ContentClass::Config | ContentClass::Unknown => 2,
            ContentClass::Test | ContentClass::Example | ContentClass::Bench => 3,
            ContentClass::Docs | ContentClass::Archive => 4,
        },
        _ => match class {
            ContentClass::Source => 0,
            ContentClass::Config => 1,
            ContentClass::Unknown => 2,
            ContentClass::Test => 3,
            ContentClass::Example | ContentClass::Bench => 4,
            ContentClass::Docs => 5,
            ContentClass::Archive => 6,
            ContentClass::Runtime => 7,
            ContentClass::Generated => 8,
        },
    }
}

fn language_for_path(file_path: &str) -> Language {
    let path = Path::new(file_path);
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(Language::from_filename)
        .or_else(|| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(Language::from_extension)
        })
        .unwrap_or(Language::Unknown)
}

fn path_depth(path: &str) -> usize {
    path.matches('/').count() + path.matches('\\').count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::metadata::MetadataStore;
    use crate::types::Chunk;

    #[test]
    fn invalid_regex_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let index_dir = tmp.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let result = search_regex(
            &index_dir,
            "[invalid",
            10,
            false,
            2,
            &SearchFilters::default(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid regex"));
    }

    #[test]
    fn source_files_are_scanned_before_docs() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(index_dir.join("docs")).unwrap_or(());
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        std::fs::create_dir_all(repo_root.join("docs")).unwrap();
        std::fs::write(
            repo_root.join("src/hotkeys.ts"),
            "export const keybind = true;\n",
        )
        .unwrap();
        std::fs::write(repo_root.join("docs/hotkeys.md"), "keybind docs\n").unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "src:0".to_string(),
                    file_path: "src/hotkeys.ts".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "export const keybind = true;".to_string(),
                    language: Language::TypeScript,
                    symbol_type: None,
                    symbol_name: None,
                },
                Chunk {
                    id: "docs:0".to_string(),
                    file_path: "docs/hotkeys.md".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "keybind docs".to_string(),
                    language: Language::Markdown,
                    symbol_type: None,
                    symbol_name: None,
                },
            ])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "keybind",
            5,
            false,
            0,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results[0].file_path, "src/hotkeys.ts");
    }

    #[test]
    fn minified_matches_return_bounded_snippets() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(repo_root.join("dist")).unwrap();
        std::fs::create_dir_all(&index_dir).unwrap();

        let content = format!(
            "var a=0;{}targetSymbol();{}",
            "x=1;".repeat(400),
            "y=2;".repeat(400)
        );
        std::fs::write(repo_root.join("dist/app.min.js"), &content).unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "dist:0".to_string(),
                file_path: "dist/app.min.js".to_string(),
                line_start: 1,
                line_end: 1,
                content: content.clone(),
                language: Language::JavaScript,
                symbol_type: None,
                symbol_name: None,
            }])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "targetSymbol",
            5,
            false,
            0,
            &SearchFilters {
                include_generated: Some(true),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].content.len() < content.len());
        assert!(results[0].content.contains("targetSymbol"));
    }
}
