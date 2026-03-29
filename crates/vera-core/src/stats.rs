//! Index statistics collection.
//!
//! Provides functions to collect and report statistics about the Vera index,
//! including file count, chunk count, index size on disk, and language breakdown.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::indexing::index_dir;
use crate::storage::metadata::MetadataStore;

/// Architecture overview of an indexed repository.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectOverview {
    /// Total files in the index.
    pub file_count: u64,
    /// Total chunks (symbols/blocks).
    pub chunk_count: u64,
    /// Approximate total lines of code.
    pub total_lines: u64,
    /// Index size on disk.
    pub index_size_human: String,
    /// Languages with file counts, sorted by file count descending.
    pub languages: Vec<LanguageOverview>,
    /// Top-level directories with file counts.
    pub top_directories: Vec<DirectoryStat>,
    /// Symbol type breakdown (function, struct, class, etc.).
    pub symbol_types: Vec<SymbolTypeStat>,
    /// Likely entry point files (main.*, index.*, app.*, etc.).
    pub entry_points: Vec<String>,
    /// Files with the most chunks (complexity hotspots).
    pub hotspots: Vec<HotspotFile>,
    /// Detected project conventions (frameworks, patterns, config files).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub conventions: Vec<String>,
}

/// Language info for the overview.
#[derive(Debug, Clone, Serialize)]
pub struct LanguageOverview {
    pub language: String,
    pub files: u64,
    pub chunks: u64,
}

/// A top-level directory with its file count.
#[derive(Debug, Clone, Serialize)]
pub struct DirectoryStat {
    pub directory: String,
    pub files: u64,
}

/// Symbol type with count.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolTypeStat {
    pub symbol_type: String,
    pub count: u64,
}

/// A file identified as a complexity hotspot.
#[derive(Debug, Clone, Serialize)]
pub struct HotspotFile {
    pub file_path: String,
    pub chunks: u64,
}

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

/// Collect an architecture overview of the indexed repository.
///
/// Returns a high-level summary: languages, directories, entry points,
/// symbol types, and complexity hotspots. Designed for agent onboarding.
pub fn collect_overview(repo_path: &Path) -> Result<ProjectOverview> {
    let idx_dir = index_dir(repo_path);

    if !idx_dir.exists() {
        anyhow::bail!(
            "no index found at: {}\nRun `vera index <path>` first to create an index.",
            idx_dir.display()
        );
    }

    let metadata_path = idx_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path).context("failed to open metadata store")?;

    let file_count = store.file_count()?;
    let chunk_count = store.chunk_count()?;
    let total_lines = store.total_lines()?;
    let index_size_bytes = compute_dir_size(&idx_dir)?;
    let index_size_human = format_bytes(index_size_bytes);

    // Merge chunk stats and file counts by language.
    let chunk_stats = store.language_stats()?;
    let file_stats = store.language_file_counts()?;
    let file_map: std::collections::HashMap<&str, u64> =
        file_stats.iter().map(|(l, c)| (l.as_str(), *c)).collect();
    let languages = chunk_stats
        .iter()
        .map(|(lang, chunks)| LanguageOverview {
            language: lang.clone(),
            files: file_map.get(lang.as_str()).copied().unwrap_or(0),
            chunks: *chunks,
        })
        .collect();

    let top_directories = store
        .top_directories(15)?
        .into_iter()
        .map(|(directory, files)| DirectoryStat { directory, files })
        .collect();

    let symbol_types = store
        .symbol_type_stats()?
        .into_iter()
        .map(|(symbol_type, count)| SymbolTypeStat { symbol_type, count })
        .collect();

    let entry_points = store.entry_points()?;

    let hotspots = store
        .hotspot_files(10)?
        .into_iter()
        .map(|(file_path, chunks)| HotspotFile { file_path, chunks })
        .collect();

    let conventions = detect_conventions(&store)?;

    Ok(ProjectOverview {
        file_count,
        chunk_count,
        total_lines,
        index_size_human,
        languages,
        top_directories,
        symbol_types,
        entry_points,
        hotspots,
        conventions,
    })
}

/// Detect project conventions by scanning indexed file paths for known patterns.
fn detect_conventions(store: &MetadataStore) -> Result<Vec<String>> {
    let files = store.indexed_files()?;
    let mut conventions = Vec::new();

    let indicators: &[(&[&str], &str)] = &[
        (&["Cargo.toml"], "Rust/Cargo project"),
        (&["package.json"], "Node.js/npm project"),
        (
            &["pyproject.toml", "setup.py", "setup.cfg"],
            "Python project",
        ),
        (&["go.mod"], "Go module"),
        (
            &["pom.xml", "build.gradle", "build.gradle.kts"],
            "Java/JVM project",
        ),
        (&["Gemfile"], "Ruby/Bundler project"),
        (
            &["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "Docker containerization",
        ),
        (&[".github/workflows"], "GitHub Actions CI"),
        (&[".gitlab-ci.yml"], "GitLab CI"),
        (&["Makefile"], "Make build system"),
        (&["tsconfig.json"], "TypeScript project"),
        (
            &[
                ".eslintrc",
                ".eslintrc.json",
                ".eslintrc.js",
                "eslint.config",
            ],
            "ESLint linting",
        ),
        (&[".prettierrc", "prettier.config"], "Prettier formatting"),
        (&["jest.config", "vitest.config"], "JS test framework"),
        (&["next.config"], "Next.js framework"),
        (&["nuxt.config"], "Nuxt.js framework"),
        (&["vite.config"], "Vite build tool"),
        (&["webpack.config"], "Webpack bundler"),
        (&["tailwind.config"], "Tailwind CSS"),
        (&[".env", ".env.example"], "Environment variable config"),
        (&["terraform"], "Terraform infrastructure"),
        (&["k8s", "kubernetes", "helm"], "Kubernetes deployment"),
        (&["proto", ".proto"], "Protocol Buffers"),
        (&["openapi", "swagger"], "OpenAPI/Swagger spec"),
        (&["migrations"], "Database migrations"),
        (&["prisma"], "Prisma ORM"),
        (&[".storybook"], "Storybook UI"),
    ];

    for (patterns, label) in indicators {
        let found = patterns.iter().any(|pat| {
            files.iter().any(|f| {
                let lower = f.to_ascii_lowercase();
                // Match as filename or path component.
                lower.contains(pat)
            })
        });
        if found {
            conventions.push((*label).to_string());
        }
    }

    Ok(conventions)
}

// ── Call graph queries ───────────────────────────────────────────────

pub use crate::storage::metadata::{CalleeRef, CallerRef, DeadSymbol};

/// Open the metadata store for a repo, or error if no index exists.
fn open_metadata(repo_path: &Path) -> Result<MetadataStore> {
    let idx_dir = index_dir(repo_path);
    if !idx_dir.exists() {
        anyhow::bail!(
            "no index found at: {}\nRun `vera index <path>` first.",
            idx_dir.display()
        );
    }
    MetadataStore::open(&idx_dir.join("metadata.db")).context("failed to open metadata store")
}

/// Find all call sites that reference a given symbol name.
pub fn find_callers(repo_path: &Path, symbol: &str) -> Result<Vec<CallerRef>> {
    open_metadata(repo_path)?.find_callers(symbol)
}

/// Find all symbols called by a given symbol.
pub fn find_callees(repo_path: &Path, symbol: &str) -> Result<Vec<CalleeRef>> {
    open_metadata(repo_path)?.find_callees(symbol)
}

/// Find defined symbols with zero callers (potential dead code).
pub fn find_dead_symbols(repo_path: &Path) -> Result<Vec<DeadSymbol>> {
    open_metadata(repo_path)?.find_dead_symbols()
}

// ── Helpers ─────────────────────────────────────────────────────────

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
