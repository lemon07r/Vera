//! Source code parsing using tree-sitter.
//!
//! This module is responsible for:
//! - Loading tree-sitter grammars for supported languages
//! - Parsing source files into ASTs
//! - Extracting symbol-level chunks (functions, classes, structs, etc.)
//! - Tier 0 fallback chunking for unsupported languages
//!
//! # Architecture
//!
//! - [`languages`] — Grammar loading and language detection
//! - [`extractor`] — AST node extraction rules per language
//! - [`chunker`] — Symbol-to-chunk conversion and large symbol splitting

pub mod chunker;
pub mod extractor;
pub mod languages;
pub mod references;
pub mod signatures;
pub mod sphinx;

use anyhow::{Context, Result};
use tree_sitter::Parser;

use crate::config::IndexingConfig;
use crate::types::{Chunk, Language};

/// Parse a source file and produce both chunks and call-site references in a
/// single pass, avoiding redundant tree-sitter parsing and symbol extraction.
///
/// For languages with tree-sitter support, the file is parsed once and the
/// resulting AST is reused for symbol extraction, chunking, and reference
/// collection. For other languages, falls back to line-based chunking with
/// no references.
///
/// # Errors
/// Returns an error if tree-sitter parsing fails for a supported language.
pub fn parse_file(
    source: &str,
    file_path: &str,
    language: Language,
    config: &IndexingConfig,
) -> Result<(Vec<Chunk>, Vec<references::RawReference>)> {
    // Special-case formats that don't use standard symbol extraction.
    if language == Language::Markdown {
        let chunks = chunker::markdown_section_chunks(source, file_path);
        return Ok((
            chunker::split_oversized_chunks(chunks, config.max_chunk_bytes),
            Vec::new(),
        ));
    }
    if language == Language::Rst {
        let chunks = parse_rst_section_chunks(source, file_path)?;
        return Ok((
            chunker::split_oversized_chunks(chunks, config.max_chunk_bytes),
            Vec::new(),
        ));
    }
    if language.prefers_file_chunking() {
        let chunks = chunker::whole_file_chunk(source, file_path, language);
        return Ok((
            chunker::split_oversized_chunks(chunks, config.max_chunk_bytes),
            Vec::new(),
        ));
    }

    let grammar = match languages::tree_sitter_grammar(language) {
        Some(g) => g,
        None => {
            let chunks = chunker::tier0_line_chunks(source, file_path, language);
            return Ok((
                chunker::split_oversized_chunks(chunks, config.max_chunk_bytes),
                Vec::new(),
            ));
        }
    };

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .context("failed to load tree-sitter grammar")?;

    let tree = parser
        .parse(source, None)
        .context("tree-sitter parsing returned None")?;

    // Single symbol extraction pass reused for both chunking and references.
    let symbols = extractor::extract_symbols(&tree, source.as_bytes(), language);
    let refs =
        references::extract_references_with_symbols(&tree, source.as_bytes(), language, &symbols);
    let chunks = chunker::chunks_from_symbols(&symbols, source, file_path, language, config);

    let chunks = if chunks.is_empty() && !source.trim().is_empty() {
        chunker::tier0_line_chunks(source, file_path, language)
    } else {
        chunks
    };

    Ok((
        chunker::split_oversized_chunks(chunks, config.max_chunk_bytes),
        refs,
    ))
}

/// Parse a source file and produce chunks (without references).
///
/// Convenience wrapper around [`parse_file`] for callers that only need chunks.
pub fn parse_and_chunk(
    source: &str,
    file_path: &str,
    language: Language,
    config: &IndexingConfig,
) -> Result<Vec<Chunk>> {
    let (chunks, _refs) = parse_file(source, file_path, language, config)?;
    Ok(chunks)
}

fn parse_rst_section_chunks(source: &str, file_path: &str) -> Result<Vec<Chunk>> {
    let grammar = languages::tree_sitter_grammar(Language::Rst)
        .context("missing tree-sitter grammar for reStructuredText")?;

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .context("failed to load reStructuredText grammar")?;

    let tree = parser
        .parse(source, None)
        .context("tree-sitter parsing returned None")?;

    let headings = extractor::extract_rst_section_titles(&tree, source.as_bytes());
    if headings.is_empty() {
        return Ok(chunker::tier0_line_chunks(source, file_path, Language::Rst));
    }

    Ok(chunker::rst_section_chunks(source, file_path, &headings))
}

/// Parse a source file and extract call-site references only.
///
/// Only works for languages with tree-sitter grammars. Returns an empty
/// vec for unsupported languages or parse failures.
pub fn parse_and_extract_references(
    source: &str,
    language: Language,
) -> Vec<references::RawReference> {
    let grammar = match languages::tree_sitter_grammar(language) {
        Some(g) => g,
        None => return Vec::new(),
    };
    let mut parser = Parser::new();
    if parser.set_language(&grammar).is_err() {
        return Vec::new();
    }
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    references::extract_references(&tree, source.as_bytes(), language)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod metadata_tests;
