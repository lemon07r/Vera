//! Symbol-aware chunking and Tier 0 fallback.
//!
//! Converts extracted AST symbols into [`Chunk`]s. Handles:
//! - Symbol → chunk mapping with metadata
//! - Large symbol splitting (>configured threshold)
//! - Gap chunks for inter-symbol code (imports, module-level statements)
//! - Tier 0 fallback: sliding-window line-based chunking for unknown languages

use crate::config::IndexingConfig;
use crate::types::{Chunk, Language, SymbolType};

use super::extractor::RawSymbol;

/// Default sliding-window size for Tier 0 fallback (lines).
const TIER0_WINDOW_SIZE: u32 = 50;
/// Default overlap for Tier 0 sliding-window (lines).
const TIER0_OVERLAP: u32 = 10;
/// Minimum lines for a symbol to be kept as a chunk (skip trivial ones).
const MIN_SYMBOL_LINES: u32 = 1;
/// Default maximum embedding text size in bytes before chunk fallback splitting.
pub const DEFAULT_MAX_EMBEDDING_TEXT_BYTES: usize = 16_384;

/// Create chunks from extracted symbols (Tier 1A: symbol-aware chunking).
///
/// Produces one chunk per symbol. Large symbols exceeding `max_chunk_lines`
/// are split into sub-chunks with no content gaps. Inter-symbol gaps
/// (imports, blank lines, module-level code) are captured as gap chunks.
pub fn chunks_from_symbols(
    symbols: &[RawSymbol],
    source: &str,
    file_path: &str,
    language: Language,
    config: &IndexingConfig,
) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len() as u32;
    let mut chunks = Vec::new();
    let mut chunk_index: u32 = 0;

    // Track coverage to identify gaps
    let mut covered_end_row: u32 = 0;

    for symbol in symbols {
        let sym_start = symbol.start_row as u32;
        let sym_end = symbol.end_row as u32;
        let sym_lines = sym_end.saturating_sub(sym_start) + 1;

        // Skip trivially small symbols (e.g., single-line forward declarations)
        if sym_lines < MIN_SYMBOL_LINES {
            continue;
        }

        // Capture gap before this symbol (imports, blank lines, etc.)
        if sym_start > covered_end_row {
            let gap_content = join_lines(&lines, covered_end_row, sym_start.saturating_sub(1));
            if !gap_content.trim().is_empty() {
                chunks.push(Chunk {
                    id: format!("{file_path}:{chunk_index}"),
                    file_path: file_path.to_string(),
                    line_start: covered_end_row + 1, // 1-based
                    line_end: sym_start,             // 1-based
                    content: gap_content,
                    language,
                    symbol_type: Some(SymbolType::Block),
                    symbol_name: None,
                });
                chunk_index += 1;
            }
        }

        // Split large symbols into sub-chunks
        if sym_lines > config.max_chunk_lines {
            let sub_chunks = split_large_symbol(
                symbol,
                &lines,
                file_path,
                language,
                config.max_chunk_lines,
                &mut chunk_index,
            );
            chunks.extend(sub_chunks);
        } else {
            let content = join_lines(&lines, sym_start, sym_end);
            chunks.push(Chunk {
                id: format!("{file_path}:{chunk_index}"),
                file_path: file_path.to_string(),
                line_start: sym_start + 1, // 1-based
                line_end: sym_end + 1,     // 1-based
                content,
                language,
                symbol_type: Some(symbol.symbol_type),
                symbol_name: symbol.name.clone(),
            });
            chunk_index += 1;
        }

        covered_end_row = sym_end + 1;
    }

    // Trailing gap after last symbol
    if covered_end_row < total_lines {
        let gap_content = join_lines(&lines, covered_end_row, total_lines.saturating_sub(1));
        if !gap_content.trim().is_empty() {
            chunks.push(Chunk {
                id: format!("{file_path}:{chunk_index}"),
                file_path: file_path.to_string(),
                line_start: covered_end_row + 1, // 1-based
                line_end: total_lines,           // 1-based
                content: gap_content,
                language,
                symbol_type: Some(SymbolType::Block),
                symbol_name: None,
            });
        }
    }

    chunks
}

/// Create a single whole-file chunk.
///
/// Used for config and document-like files where the filename and full file
/// context are more important than symbol-level segmentation.
pub fn whole_file_chunk(source: &str, file_path: &str, language: Language) -> Vec<Chunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }

    vec![Chunk {
        id: format!("{file_path}:0"),
        file_path: file_path.to_string(),
        line_start: 1,
        line_end: source.lines().count().max(1) as u32,
        content: source.to_string(),
        language,
        symbol_type: Some(SymbolType::Block),
        symbol_name: Some(file_name(file_path).to_string()),
    }]
}

/// Split a large symbol into sub-chunks of at most `max_lines` lines.
///
/// Uses smart boundary detection: prefers splitting after closing braces (`}`),
/// semicolons, or blank lines rather than at arbitrary line counts. Falls back
/// to the hard `max_lines` limit when no good boundary exists nearby.
fn split_large_symbol(
    symbol: &RawSymbol,
    lines: &[&str],
    file_path: &str,
    language: Language,
    max_lines: u32,
    chunk_index: &mut u32,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let start = symbol.start_row as u32;
    let end = symbol.end_row as u32;
    let mut current = start;
    let mut part = 1u32;

    while current <= end {
        let ideal_end = (current + max_lines - 1).min(end);
        let chunk_end = if ideal_end >= end {
            end
        } else {
            find_split_boundary(lines, current, ideal_end, max_lines)
        };
        let content = join_lines(lines, current, chunk_end);
        let sub_name = symbol.name.as_ref().map(|n| format!("{n} (part {part})"));

        chunks.push(Chunk {
            id: format!("{file_path}:{}", *chunk_index),
            file_path: file_path.to_string(),
            line_start: current + 1, // 1-based
            line_end: chunk_end + 1, // 1-based
            content,
            language,
            symbol_type: Some(symbol.symbol_type),
            symbol_name: sub_name,
        });

        *chunk_index += 1;
        part += 1;
        current = chunk_end + 1;
    }

    chunks
}

/// Find the best line to split at, searching backward from `ideal_end`.
///
/// Looks for (in priority order): closing brace on its own line, semicolon at
/// end of line, blank line. Searches up to 30% of `max_lines` backward. If no
/// good boundary is found, returns `ideal_end` (hard split).
fn find_split_boundary(lines: &[&str], start: u32, ideal_end: u32, max_lines: u32) -> u32 {
    let lookback = (max_lines * 3 / 10).max(3).min(ideal_end - start);
    let search_start = ideal_end.saturating_sub(lookback);

    // Pass 1: closing brace alone on a line (strongest boundary).
    for row in (search_start..=ideal_end).rev() {
        let trimmed = lines.get(row as usize).map(|l| l.trim()).unwrap_or("");
        if trimmed == "}" || trimmed == "};" || trimmed == "}," {
            return row;
        }
    }

    // Pass 2: line ending with semicolon or blank line.
    for row in (search_start..=ideal_end).rev() {
        let trimmed = lines.get(row as usize).map(|l| l.trim()).unwrap_or("");
        if trimmed.is_empty() || trimmed.ends_with(';') {
            return row;
        }
    }

    ideal_end
}

/// Split a markdown file into section-based chunks.
///
/// Each heading (# through ######) starts a new chunk. Content before the
/// first heading becomes a chunk named after the file. This produces better
/// search results than whole-file chunking because each section has a
/// focused topic and heading as its symbol name.
pub fn markdown_section_chunks(source: &str, file_path: &str) -> Vec<Chunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut chunk_index: u32 = 0;

    // Track current section
    let mut section_start: usize = 0;
    let mut section_name: Option<String> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            // Extract heading text (strip leading #s and whitespace)
            let heading = trimmed.trim_start_matches('#').trim();
            if heading.is_empty() && !trimmed.contains(' ') {
                // Not a real heading (e.g., just "###" with no text)
                continue;
            }

            // Flush previous section
            if i > section_start {
                let content = join_lines(&lines, section_start as u32, (i - 1) as u32);
                if !content.trim().is_empty() {
                    let name = section_name
                        .take()
                        .unwrap_or_else(|| file_name(file_path).to_string());
                    chunks.push(Chunk {
                        id: format!("{file_path}:{chunk_index}"),
                        file_path: file_path.to_string(),
                        line_start: section_start as u32 + 1,
                        line_end: i as u32,
                        content,
                        language: Language::Markdown,
                        symbol_type: Some(SymbolType::Block),
                        symbol_name: Some(name),
                    });
                    chunk_index += 1;
                }
            }

            section_start = i;
            section_name = Some(heading.to_string());
        }
    }

    // Flush final section
    let content = join_lines(&lines, section_start as u32, (lines.len() - 1) as u32);
    if !content.trim().is_empty() {
        let name = section_name.unwrap_or_else(|| file_name(file_path).to_string());
        chunks.push(Chunk {
            id: format!("{file_path}:{chunk_index}"),
            file_path: file_path.to_string(),
            line_start: section_start as u32 + 1,
            line_end: lines.len() as u32,
            content,
            language: Language::Markdown,
            symbol_type: Some(SymbolType::Block),
            symbol_name: Some(name),
        });
    }

    // If no headings found, fall back to single whole-file chunk
    if chunks.is_empty() {
        return whole_file_chunk(source, file_path, Language::Markdown);
    }

    chunks
}

/// Tier 0 fallback: sliding-window line-based chunking.
///
/// Used for files with no tree-sitter grammar support. Produces overlapping
/// chunks of `TIER0_WINDOW_SIZE` lines with `TIER0_OVERLAP` overlap.
pub fn tier0_line_chunks(source: &str, file_path: &str, language: Language) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    let total = lines.len() as u32;

    if total == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start: u32 = 0;
    let mut chunk_index: u32 = 0;
    let step = TIER0_WINDOW_SIZE.saturating_sub(TIER0_OVERLAP);

    while start < total {
        let end = (start + TIER0_WINDOW_SIZE - 1).min(total - 1);
        let content = join_lines(&lines, start, end);

        if !content.trim().is_empty() {
            chunks.push(Chunk {
                id: format!("{file_path}:{chunk_index}"),
                file_path: file_path.to_string(),
                line_start: start + 1, // 1-based
                line_end: end + 1,     // 1-based
                content,
                language,
                symbol_type: Some(SymbolType::Block),
                symbol_name: None,
            });
            chunk_index += 1;
        }

        // Avoid infinite loop when step is 0
        if step == 0 {
            break;
        }
        start += step;
    }

    chunks
}

/// Split chunks whose embedding text would exceed `max_bytes`.
///
/// This is a strict guard: every output chunk is adjusted to fit the limit
/// (except pathological cases where even metadata alone would exceed it).
/// Splitting uses overlapping line windows to preserve continuity.
pub fn split_oversized_chunks(chunks: Vec<Chunk>, max_bytes: usize) -> Vec<Chunk> {
    let mut result = Vec::with_capacity(chunks.len());
    let input_count = chunks.len();
    let mut split_count = 0u32;
    let mut truncated_single_line = 0u32;
    let mut max_seen = 0usize;

    for chunk in chunks {
        let text_len = crate::chunk_text::build_embedding_text(&chunk).len();
        max_seen = max_seen.max(text_len);
        if text_len <= max_bytes {
            result.push(chunk);
            continue;
        }

        let split = split_chunk_to_max_chars(&chunk, max_bytes);
        if split.len() > 1 {
            split_count += 1;
        }

        for mut sub in split {
            let mut sub_len = crate::chunk_text::build_embedding_text(&sub).len();
            if sub_len <= max_bytes {
                result.push(sub);
                continue;
            }

            // If this still overflows and has multiple lines, fall back to
            // one-line chunks so each piece can be individually constrained.
            if sub.content.lines().count() > 1 {
                for (idx, line) in sub.content.lines().enumerate() {
                    let mut line_chunk = Chunk {
                        id: format!("{}::line{}", sub.id, idx + 1),
                        file_path: sub.file_path.clone(),
                        line_start: sub.line_start + idx as u32,
                        line_end: sub.line_start + idx as u32,
                        content: line.to_string(),
                        language: sub.language,
                        symbol_type: sub.symbol_type,
                        symbol_name: sub
                            .symbol_name
                            .as_ref()
                            .map(|name| format!("{name} (line {})", idx + 1)),
                    };

                    let mut line_len = crate::chunk_text::build_embedding_text(&line_chunk).len();
                    if line_len > max_bytes {
                        truncated_single_line += 1;
                        let header_len = line_len.saturating_sub(line_chunk.content.len());
                        let keep = max_bytes.saturating_sub(header_len).max(1);
                        line_chunk.content = truncate_utf8_to_bytes(&line_chunk.content, keep);
                        line_len = crate::chunk_text::build_embedding_text(&line_chunk).len();
                        if line_len > max_bytes {
                            line_chunk.content.clear();
                        }
                    }

                    result.push(line_chunk);
                }
                continue;
            }

            // Last-resort safeguard for very long single-line chunks.
            if sub.content.lines().count() <= 1 {
                truncated_single_line += 1;
                let header_len = sub_len.saturating_sub(sub.content.len());
                let keep = max_bytes.saturating_sub(header_len).max(1);
                sub.content = truncate_utf8_to_bytes(&sub.content, keep);
                sub_len = crate::chunk_text::build_embedding_text(&sub).len();
                if sub_len > max_bytes {
                    sub.content.clear();
                }
            }

            result.push(sub);
        }
    }

    tracing::debug!(
        split_count,
        truncated_single_line,
        max_bytes,
        max_embedding_text_bytes = max_seen,
        input_chunks = input_count,
        result_chunks = result.len(),
        "chunk size check complete"
    );
    result
}

fn split_chunk_to_max_chars(chunk: &Chunk, max_chars: usize) -> Vec<Chunk> {
    let full_text_len = crate::chunk_text::build_embedding_text(chunk).len();
    if full_text_len <= max_chars {
        return vec![chunk.clone()];
    }

    let lines: Vec<&str> = chunk.content.lines().collect();
    let total_lines = lines.len() as u32;
    if total_lines <= 1 {
        return vec![chunk.clone()];
    }

    let header_len = full_text_len.saturating_sub(chunk.content.len());
    let content_budget = max_chars.saturating_sub(header_len);
    if content_budget < 64 {
        return vec![chunk.clone()];
    }

    let mut out = Vec::new();
    let mut start = 0u32;
    let overlap = 2u32;
    let mut part = 1u32;

    while start < total_lines {
        let mut end = find_end_for_content_budget(&lines, start, content_budget);
        let mut sub = make_sub_chunk(chunk, &lines, start, end, part);
        let mut sub_len = crate::chunk_text::build_embedding_text(&sub).len();

        // Strictly shrink until this piece fits.
        while sub_len > max_chars && end > start {
            end -= 1;
            sub = make_sub_chunk(chunk, &lines, start, end, part);
            sub_len = crate::chunk_text::build_embedding_text(&sub).len();
        }

        out.push(sub);

        if end + 1 >= total_lines {
            break;
        }

        let next_start = (end + 1).saturating_sub(overlap);
        start = if next_start <= start {
            start + 1
        } else {
            next_start
        };
        part += 1;
    }

    if out.is_empty() {
        vec![chunk.clone()]
    } else {
        out
    }
}

fn find_end_for_content_budget(lines: &[&str], start: u32, budget: usize) -> u32 {
    let total = lines.len() as u32;
    let mut used = 0usize;
    let mut end = start;

    while end < total {
        let add = lines[end as usize].len() + 1;
        if end > start && used + add > budget {
            break;
        }
        used += add;
        end += 1;
    }

    end.saturating_sub(1)
}

fn make_sub_chunk(chunk: &Chunk, lines: &[&str], start: u32, end: u32, part: u32) -> Chunk {
    let sub_content = lines[start as usize..=end as usize].join("\n");
    let sub_name = chunk
        .symbol_name
        .as_ref()
        .map(|name| format!("{name} (part {part})"));

    Chunk {
        id: format!("{}::part{}", chunk.id, part),
        file_path: chunk.file_path.clone(),
        line_start: chunk.line_start + start,
        line_end: chunk.line_start + end,
        content: sub_content,
        language: chunk.language,
        symbol_type: chunk.symbol_type,
        symbol_name: sub_name,
    }
}

fn truncate_utf8_to_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }

    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Join lines from `start_row` to `end_row` (inclusive, 0-based) into a string.
fn join_lines(lines: &[&str], start_row: u32, end_row: u32) -> String {
    let start = start_row as usize;
    let end = (end_row as usize).min(lines.len().saturating_sub(1));
    if start > end || start >= lines.len() {
        return String::new();
    }
    lines[start..=end].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::extractor::RawSymbol;
    use crate::types::SymbolType;

    fn default_config() -> IndexingConfig {
        IndexingConfig {
            max_chunk_lines: 200,
            ..Default::default()
        }
    }

    #[test]
    fn single_symbol_becomes_one_chunk() {
        let source = "fn hello() {\n    println!(\"hi\");\n}\n";
        let symbols = vec![RawSymbol {
            name: Some("hello".to_string()),
            symbol_type: SymbolType::Function,
            start_byte: 0,
            end_byte: source.len(),
            start_row: 0,
            end_row: 2,
        }];
        let chunks = chunks_from_symbols(
            &symbols,
            source,
            "test.rs",
            Language::Rust,
            &default_config(),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].symbol_name, Some("hello".to_string()));
        assert_eq!(chunks[0].symbol_type, Some(SymbolType::Function));
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 3);
    }

    #[test]
    fn gap_between_symbols_captured() {
        let source = "use std::io;\n\nfn hello() {\n}\n\nfn world() {\n}\n";
        let symbols = vec![
            RawSymbol {
                name: Some("hello".to_string()),
                symbol_type: SymbolType::Function,
                start_byte: 0,
                end_byte: 0,
                start_row: 2,
                end_row: 3,
            },
            RawSymbol {
                name: Some("world".to_string()),
                symbol_type: SymbolType::Function,
                start_byte: 0,
                end_byte: 0,
                start_row: 5,
                end_row: 6,
            },
        ];
        let chunks = chunks_from_symbols(
            &symbols,
            source,
            "test.rs",
            Language::Rust,
            &default_config(),
        );
        // Should have: gap (imports), hello, world
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            chunks.len()
        );
        // First chunk should be the gap (imports)
        assert_eq!(chunks[0].symbol_type, Some(SymbolType::Block));
        assert!(chunks[0].content.contains("use std::io"));
    }

    #[test]
    fn large_symbol_split_into_sub_chunks() {
        // Create a large function (10 lines, with max_chunk_lines=3)
        let mut lines = vec!["fn big() {".to_string()];
        for i in 0..8 {
            lines.push(format!("    let x{i} = {i};"));
        }
        lines.push("}".to_string());
        let source = lines.join("\n");

        let symbols = vec![RawSymbol {
            name: Some("big".to_string()),
            symbol_type: SymbolType::Function,
            start_byte: 0,
            end_byte: source.len(),
            start_row: 0,
            end_row: 9,
        }];

        let config = IndexingConfig {
            max_chunk_lines: 3,
            ..Default::default()
        };
        let chunks = chunks_from_symbols(&symbols, &source, "test.rs", Language::Rust, &config);

        // 10 lines / 3 lines per chunk = 4 sub-chunks (3+3+3+1)
        assert_eq!(
            chunks.len(),
            4,
            "expected 4 sub-chunks, got {}",
            chunks.len()
        );

        // Verify no content gaps: reconstruct and compare
        let mut all_content = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 {
                all_content.push('\n');
            }
            all_content.push_str(&chunk.content);
            // Sub-chunks should have part numbers in name
            assert!(
                chunk.symbol_name.as_ref().unwrap().contains("part"),
                "sub-chunk should have part number"
            );
        }
        assert_eq!(all_content, source);
    }

    #[test]
    fn tier0_fallback_produces_chunks() {
        let mut lines = Vec::new();
        for i in 0..120 {
            lines.push(format!("line {i}"));
        }
        let source = lines.join("\n");

        let chunks = tier0_line_chunks(&source, "data.xyz", Language::Unknown);
        // 120 lines, window=50, overlap=10, step=40
        // Chunks: [0..49], [40..89], [80..119] = 3 chunks
        assert_eq!(
            chunks.len(),
            3,
            "expected 3 tier0 chunks, got {}",
            chunks.len()
        );

        // All chunks should have Block type
        for chunk in &chunks {
            assert_eq!(chunk.symbol_type, Some(SymbolType::Block));
            assert_eq!(chunk.language, Language::Unknown);
        }

        // First chunk starts at line 1
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 50);

        // Second chunk overlaps
        assert_eq!(chunks[1].line_start, 41);
        assert_eq!(chunks[1].line_end, 90);
    }

    #[test]
    fn tier0_empty_source_no_chunks() {
        let chunks = tier0_line_chunks("", "empty.xyz", Language::Unknown);
        assert!(chunks.is_empty());
    }

    #[test]
    fn tier0_small_file_one_chunk() {
        let source = "line 1\nline 2\nline 3";
        let chunks = tier0_line_chunks(source, "small.xyz", Language::Unknown);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 3);
    }

    #[test]
    fn chunks_have_correct_file_path() {
        let source = "fn foo() {}\n";
        let symbols = vec![RawSymbol {
            name: Some("foo".to_string()),
            symbol_type: SymbolType::Function,
            start_byte: 0,
            end_byte: source.len(),
            start_row: 0,
            end_row: 0,
        }];
        let chunks = chunks_from_symbols(
            &symbols,
            source,
            "src/lib.rs",
            Language::Rust,
            &default_config(),
        );
        assert_eq!(chunks[0].file_path, "src/lib.rs");
    }

    #[test]
    fn chunks_have_unique_ids() {
        let source = "fn a() {}\nfn b() {}\n";
        let symbols = vec![
            RawSymbol {
                name: Some("a".to_string()),
                symbol_type: SymbolType::Function,
                start_byte: 0,
                end_byte: 0,
                start_row: 0,
                end_row: 0,
            },
            RawSymbol {
                name: Some("b".to_string()),
                symbol_type: SymbolType::Function,
                start_byte: 0,
                end_byte: 0,
                start_row: 1,
                end_row: 1,
            },
        ];
        let chunks = chunks_from_symbols(
            &symbols,
            source,
            "test.rs",
            Language::Rust,
            &default_config(),
        );
        let ids: Vec<_> = chunks.iter().map(|c| &c.id).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "all chunk IDs should be unique");
    }

    #[test]
    fn whole_file_chunk_uses_filename_as_symbol() {
        let chunks = whole_file_chunk("[workspace]\nmembers = []\n", "Cargo.toml", Language::Toml);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].symbol_name.as_deref(), Some("Cargo.toml"));
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 2);
    }

    fn make_chunk(content: &str) -> Chunk {
        Chunk {
            id: "test::chunk".to_string(),
            file_path: "src/test.rs".to_string(),
            line_start: 1,
            line_end: content.lines().count().max(1) as u32,
            content: content.to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("test_symbol".to_string()),
        }
    }

    #[test]
    fn split_oversized_chunks_multi_line_within_limit() {
        let content = (0..60)
            .map(|i| format!("line {i} with repeated content for split testing"))
            .collect::<Vec<_>>()
            .join("\n");
        let max_bytes = 280;
        let chunks = split_oversized_chunks(vec![make_chunk(&content)], max_bytes);
        assert!(chunks.len() > 1);
        for chunk in chunks {
            let len = crate::chunk_text::build_embedding_text(&chunk).len();
            assert!(
                len <= max_bytes,
                "embedding text length {len} exceeded {max_bytes}"
            );
        }
    }

    #[test]
    fn split_oversized_chunks_single_line_fallback_within_limit() {
        let max_bytes = 180;
        let chunk = make_chunk(&"x".repeat(8_000));
        let chunks = split_oversized_chunks(vec![chunk], max_bytes);
        assert!(!chunks.is_empty());
        for chunk in chunks {
            let len = crate::chunk_text::build_embedding_text(&chunk).len();
            assert!(
                len <= max_bytes,
                "single-line fallback length {len} exceeded {max_bytes}"
            );
        }
    }

    #[test]
    fn split_oversized_chunks_utf8_boundary_within_limit() {
        let max_bytes = 140;
        let chunk = make_chunk(&"😀".repeat(1500));
        let chunks = split_oversized_chunks(vec![chunk], max_bytes);
        assert!(!chunks.is_empty());
        for chunk in chunks {
            let text = crate::chunk_text::build_embedding_text(&chunk);
            assert!(text.len() <= max_bytes);
            assert!(text.is_char_boundary(text.len()));
        }
    }
}
