//! Index statistics collection.
//!
//! Provides functions to collect and report statistics about the Vera index,
//! including file count, chunk count, index size on disk, and language breakdown.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::indexing::index_dir;
use crate::storage::metadata::MetadataStore;

/// Complete statistics about an indexed repository.
#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    /// Number of distinct source files in the index.
    pub file_count: u64,
    /// Total number of chunks (symbols/blocks) in the index.
    pub chunk_count: u64,
    /// Total index size on disk in bytes.
    pub index_size_bytes: u64,
    /// Human-readable index size (e.g., "12.3 MB").
    pub index_size_human: String,
    /// Language breakdown: language name -> chunk count.
    pub languages: Vec<LanguageStat>,
}

/// Statistics for a single programming language.
#[derive(Debug, Clone, Serialize)]
pub struct LanguageStat {
    /// Language name (e.g., "rust", "python").
    pub language: String,
    /// Number of chunks in this language.
    pub chunk_count: u64,
    /// Percentage of total chunks (0.0–100.0).
    pub percentage: f64,
}

/// Collect statistics about the index stored at the given repository path.
///
/// # Arguments
/// - `repo_path` — Path to the repository root (index lives in `.vera/`).
///
/// # Errors
/// Returns an error if the index doesn't exist or can't be read.
pub fn collect_stats(repo_path: &Path) -> Result<IndexStats> {
    let idx_dir = index_dir(repo_path);

    if !idx_dir.exists() {
        anyhow::bail!(
            "no index found at: {}\nRun `vera index <path>` first to create an index.",
            idx_dir.display()
        );
    }

    // Open metadata store.
    let metadata_path = idx_dir.join("metadata.db");
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;

    // Collect counts.
    let file_count = metadata_store
        .file_count()
        .context("failed to get file count")?;
    let chunk_count = metadata_store
        .chunk_count()
        .context("failed to get chunk count")?;

    // Compute index size on disk.
    let index_size_bytes = compute_dir_size(&idx_dir).context("failed to compute index size")?;
    let index_size_human = format_bytes(index_size_bytes);

    // Collect language breakdown.
    let raw_lang_stats = metadata_store
        .language_stats()
        .context("failed to get language stats")?;

    let languages: Vec<LanguageStat> = raw_lang_stats
        .into_iter()
        .map(|(language, count)| {
            let percentage = if chunk_count > 0 {
                (count as f64 / chunk_count as f64) * 100.0
            } else {
                0.0
            };
            LanguageStat {
                language,
                chunk_count: count,
                percentage,
            }
        })
        .collect();

    Ok(IndexStats {
        file_count,
        chunk_count,
        index_size_bytes,
        index_size_human,
        languages,
    })
}

/// Recursively compute the total size of files in a directory.
fn compute_dir_size(dir: &Path) -> Result<u64> {
    let mut total = 0u64;
    if dir.is_file() {
        return Ok(dir.metadata().map(|m| m.len()).unwrap_or(0));
    }
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?;
    for entry in entries {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if path.is_dir() {
            total += compute_dir_size(&path)?;
        } else {
            total += path.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    Ok(total)
}

/// Format a byte count into a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn collect_stats_missing_index() {
        let dir = std::env::temp_dir().join("vera-stats-test-missing");
        let result = collect_stats(&dir);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no index found"), "error was: {err}");
    }
}
