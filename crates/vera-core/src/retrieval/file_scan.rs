use std::path::Path;

use crate::corpus::ContentClass;
use crate::types::{Chunk, Language, SearchFilters, SymbolType};

pub(crate) fn language_for_path(file_path: &str) -> Language {
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

pub(crate) fn allows_class(filters: &SearchFilters, class: ContentClass) -> bool {
    match filters.scope {
        Some(scope) => {
            crate::corpus::matches_scope(class, scope, filters.include_generated.unwrap_or(true))
        }
        None => true,
    }
}

pub(crate) fn file_scan_priority(class: ContentClass, filters: &SearchFilters) -> u8 {
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

pub(crate) fn smallest_symbol_chunk_for_line(chunks: &[Chunk], line: u32) -> Option<&Chunk> {
    chunks
        .iter()
        .filter(|chunk| chunk.line_start <= line && line <= chunk.line_end)
        .filter(|chunk| chunk.symbol_type.is_some() || chunk.symbol_name.is_some())
        .min_by_key(|chunk| {
            (
                chunk.line_end.saturating_sub(chunk.line_start),
                chunk.line_start,
                chunk.line_end,
            )
        })
}

pub(crate) fn symbol_for_line(
    chunks: Option<&[Chunk]>,
    line: u32,
) -> (Option<String>, Option<SymbolType>) {
    chunks
        .and_then(|chunks| {
            smallest_symbol_chunk_for_line(chunks, line)
                .map(|chunk| (chunk.symbol_name.clone(), chunk.symbol_type))
        })
        .unwrap_or((None, None))
}

pub(crate) fn bounded_byte_snippet(
    content: &str,
    start_byte: usize,
    end_byte: usize,
    window: usize,
) -> (String, u32, u32) {
    let start = clamp_char_boundary(content, start_byte.saturating_sub(window));
    let end = clamp_char_boundary(content, (end_byte + window).min(content.len()));
    let mut snippet = content[start..end].to_string();

    if start > 0 {
        snippet.insert_str(0, "...");
    }
    if end < content.len() {
        snippet.push_str("...");
    }

    let line_start = byte_to_line(content, start_byte);
    let line_end = byte_to_line(content, end_byte);
    (snippet, line_start, line_end.max(line_start))
}

pub(crate) fn line_context_snippet(
    content: &str,
    line: u32,
    context_lines: usize,
) -> (String, u32, u32) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return (String::new(), line, line);
    }

    let idx = line.saturating_sub(1) as usize;
    let start = idx.saturating_sub(context_lines).min(lines.len() - 1);
    let end = (idx + context_lines + 1).min(lines.len());
    (lines[start..end].join("\n"), (start + 1) as u32, end as u32)
}

fn clamp_char_boundary(content: &str, mut idx: usize) -> usize {
    while idx > 0 && !content.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

pub(crate) fn byte_to_line(content: &str, byte_idx: usize) -> u32 {
    content[..byte_idx.min(content.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count() as u32
        + 1
}
