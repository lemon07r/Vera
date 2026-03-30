//! File discovery: walk directory tree, respect .gitignore, exclude defaults.
//!
//! This module discovers source files for indexing by:
//! - Walking the directory tree using the `ignore` crate (respects .gitignore)
//! - Applying default exclusion patterns (.git, node_modules, target, etc.)
//! - Detecting and skipping binary files (content heuristics + extension checks)
//! - Handling large files (skip above configurable threshold)
//!
//! # Architecture
//!
//! Uses the `ignore` crate which natively supports:
//! - `.gitignore` patterns (hierarchical, all levels)
//! - Global gitignore
//! - `.ignore` files
//! - Custom override patterns

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use tracing::{debug, warn};

use crate::config::IndexingConfig;

/// A discovered source file ready for indexing.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    /// Absolute path to the file.
    pub absolute_path: PathBuf,
    /// Repository-relative path (for chunk metadata).
    pub relative_path: String,
    /// File size in bytes.
    pub size: u64,
}

/// Result of the file discovery process.
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Files that passed all filters and are ready to index.
    pub files: Vec<DiscoveredFile>,
    /// Number of files skipped because they were binary.
    pub binary_skipped: usize,
    /// Number of files skipped because they exceeded the size threshold.
    pub large_skipped: usize,
    /// Relative paths and sizes (bytes) of files skipped because they exceeded the size threshold.
    pub large_skipped_paths: Vec<(String, u64)>,
    /// Number of files skipped due to read errors (permissions, etc.).
    pub error_skipped: usize,
}

/// Known binary file extensions (skip without content inspection).
///
/// Categories: compiled objects, archives, images, audio/video, fonts,
/// documents, databases, JVM bytecode, .NET, lock files, compiled Python, WASM.
#[rustfmt::skip]
const BINARY_EXTENSIONS: &[&str] = &[
    "o", "obj", "a", "lib", "so", "dylib", "dll", "exe", "com",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "zst",
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp", "tiff", "tif",
    "mp3", "mp4", "avi", "mov", "wav", "flac", "ogg", "webm", "mkv",
    "ttf", "otf", "woff", "woff2", "eot",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "db", "sqlite", "sqlite3",
    "class", "jar", "war", "ear",
    "pdb", "nupkg",
    "lock",
    "pyc", "pyo",
    "wasm",
    "bin", "dat", "pak",
];

/// Discover source files in a directory tree.
///
/// Walks the directory respecting .gitignore patterns and default exclusions.
/// Skips binary files and files exceeding the size threshold.
pub fn discover_files(root: &Path, config: &IndexingConfig) -> Result<DiscoveryResult> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve path: {}", root.display()))?;

    // Determine ignore file strategy based on .veraignore presence.
    let veraignore_path = root.join(".veraignore");
    let (use_gitignore, use_veraignore) = if config.no_ignore {
        (false, false)
    } else if veraignore_path.exists() {
        let content = std::fs::read_to_string(&veraignore_path)
            .with_context(|| format!("failed to read {}", veraignore_path.display()))?;
        let has_include = content.lines().any(|l| l.trim() == "#include .gitignore");
        (has_include, true)
    } else {
        (true, false)
    };

    let mut walker = WalkBuilder::new(&root);
    walker
        .hidden(false)
        .git_ignore(use_gitignore)
        .git_global(use_gitignore)
        .git_exclude(use_gitignore)
        .require_git(false)
        .ignore(!config.no_ignore);

    if use_veraignore {
        walker.add_custom_ignore_filename(".veraignore");
    }

    // Add default directory exclusions and CLI --exclude patterns as overrides.
    let mut overrides = ignore::overrides::OverrideBuilder::new(&root);
    if !config.no_default_excludes {
        for pattern in &config.default_excludes {
            // Exclude both files and directories matching the pattern
            let file_glob = format!("!{pattern}");
            overrides
                .add(&file_glob)
                .with_context(|| format!("invalid exclusion pattern: {pattern}"))?;

            let dir_glob = if pattern.ends_with('/') {
                format!("!{pattern}")
            } else {
                format!("!{pattern}/")
            };
            overrides
                .add(&dir_glob)
                .with_context(|| format!("invalid exclusion pattern: {pattern}"))?;
        }
        // Always exclude .veraignore itself from indexing.
        overrides.add("!.veraignore")?;
    }
    for pattern in &config.extra_excludes {
        overrides
            .add(&format!("!{pattern}"))
            .with_context(|| format!("invalid --exclude pattern: {pattern}"))?;
    }
    let overrides = overrides
        .build()
        .context("failed to build override patterns")?;
    walker.overrides(overrides);

    let mut files = Vec::new();
    let mut binary_skipped = 0usize;
    let mut large_skipped = 0usize;
    let mut large_skipped_paths = Vec::new();
    let mut error_skipped = 0usize;

    for entry in walker.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                warn!("file discovery error: {err}");
                error_skipped += 1;
                continue;
            }
        };

        // Skip directories — we only want files.
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Get file metadata for size check.
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(err) => {
                warn!("cannot read metadata for {}: {err}", path.display());
                error_skipped += 1;
                continue;
            }
        };

        let size = metadata.len();

        // Skip large files.
        if size > config.max_file_size_bytes {
            debug!(
                "skipping large file: {} ({} bytes > {} max)",
                path.display(),
                size,
                config.max_file_size_bytes
            );
            let rel = path
                .strip_prefix(&root)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            large_skipped_paths.push((rel, size));
            large_skipped += 1;
            continue;
        }

        // Skip empty files.
        if size == 0 {
            continue;
        }

        // Skip binary files by extension.
        if is_binary_extension(path) {
            debug!("skipping binary (extension): {}", path.display());
            binary_skipped += 1;
            continue;
        }

        // Skip binary files by content detection (read first 8KB).
        match is_binary_content(path) {
            Ok(true) => {
                debug!("skipping binary (content): {}", path.display());
                binary_skipped += 1;
                continue;
            }
            Ok(false) => {}
            Err(err) => {
                warn!(
                    "cannot read file for binary check: {} — {err}",
                    path.display()
                );
                error_skipped += 1;
                continue;
            }
        }

        // Compute repository-relative path.
        let relative_path = match path.strip_prefix(&root) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => path.to_string_lossy().to_string(),
        };

        files.push(DiscoveredFile {
            absolute_path: path.to_path_buf(),
            relative_path,
            size,
        });
    }

    // Sort for deterministic output.
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    Ok(DiscoveryResult {
        files,
        binary_skipped,
        large_skipped,
        large_skipped_paths,
        error_skipped,
    })
}

/// Check if a file has a known binary extension.
fn is_binary_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            let lower = ext.to_lowercase();
            BINARY_EXTENSIONS.contains(&lower.as_str())
        })
}

/// Check if a file contains binary content by reading the first 8KB.
///
/// Uses the null-byte heuristic: if any null bytes are found in the
/// first 8KB, the file is considered binary.
fn is_binary_content(path: &Path) -> Result<bool> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("cannot open for binary check: {}", path.display()))?;

    let mut buf = [0u8; 8192];
    let n = file.read(&mut buf)?;

    // Null byte detection: any \0 in the sample means binary.
    Ok(buf[..n].contains(&0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn default_config() -> IndexingConfig {
        IndexingConfig::default()
    }

    #[test]
    fn discovers_source_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.py"), "def hello(): pass").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 2);

        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"lib.py"));
        assert!(names.contains(&"main.rs"));
    }

    #[test]
    fn respects_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive data").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignored file should be excluded"
        );
    }

    #[test]
    fn default_exclusions_active() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        // Create excluded directories with files.
        let nm = dir.path().join("node_modules");
        fs::create_dir_all(&nm).unwrap();
        fs::write(nm.join("dep.js"), "module.exports = {}").unwrap();

        let git = dir.path().join(".git");
        fs::create_dir_all(&git).unwrap();
        fs::write(git.join("config"), "[core]").unwrap();

        let target = dir.path().join("target");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("out.rs"), "compiled").unwrap();

        let pycache = dir.path().join("__pycache__");
        fs::create_dir_all(&pycache).unwrap();
        fs::write(pycache.join("mod.pyc"), "bytecode").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(names.contains(&"main.rs"), "main.rs should be included");
        assert!(
            !names.iter().any(|n| n.starts_with("node_modules")),
            "node_modules should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with(".git")),
            ".git should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("target")),
            "target should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("__pycache__")),
            "__pycache__ should be excluded"
        );
    }

    #[test]
    fn skips_binary_by_extension() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("image.png"), "not really png").unwrap();
        fs::write(dir.path().join("program.exe"), "not really exe").unwrap();
        fs::write(dir.path().join("archive.zip"), "not really zip").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
        assert_eq!(result.binary_skipped, 3);
    }

    #[test]
    fn skips_binary_by_content() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        // Write a file with null bytes (binary content).
        let binary_content = b"some text\x00more text";
        fs::write(dir.path().join("data.txt"), binary_content).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
        assert_eq!(result.binary_skipped, 1);
    }

    #[test]
    fn skips_large_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("small.rs"), "fn main() {}").unwrap();

        // Create a file larger than the default threshold (1MB).
        let large_content = "x".repeat(1_100_000);
        fs::write(dir.path().join("huge.rs"), large_content).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "small.rs");
        assert_eq!(result.large_skipped, 1);
    }

    #[test]
    fn configurable_size_threshold() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("small.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("medium.rs"), "x".repeat(500)).unwrap();

        let config = IndexingConfig {
            max_file_size_bytes: 100,
            ..Default::default()
        };

        let result = discover_files(dir.path(), &config).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "small.rs");
        assert_eq!(result.large_skipped, 1);
    }

    #[test]
    fn skips_empty_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("empty.rs"), "").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
    }

    #[test]
    fn discovers_nested_files() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("src").join("utils");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("helper.rs"), "fn help() {}").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 2);

        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(paths.contains(&"main.rs"));
        assert!(paths.contains(&"src/utils/helper.rs"));
    }

    #[test]
    fn binary_extension_detection() {
        assert!(is_binary_extension(Path::new("file.png")));
        assert!(is_binary_extension(Path::new("file.PNG")));
        assert!(is_binary_extension(Path::new("file.exe")));
        assert!(is_binary_extension(Path::new("file.zip")));
        assert!(is_binary_extension(Path::new("file.wasm")));
        assert!(!is_binary_extension(Path::new("file.rs")));
        assert!(!is_binary_extension(Path::new("file.py")));
        assert!(!is_binary_extension(Path::new("file.ts")));
        assert!(!is_binary_extension(Path::new("no_extension")));
    }

    #[test]
    fn result_is_deterministically_sorted() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("z.rs"), "fn z() {}").unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("m.rs"), "fn m() {}").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert_eq!(paths, vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[test]
    fn venv_and_dist_excluded() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("app.py"), "def main(): pass").unwrap();

        let venv = dir.path().join(".venv");
        fs::create_dir_all(&venv).unwrap();
        fs::write(venv.join("activate"), "#!/bin/bash").unwrap();

        let dist = dir.path().join("dist");
        fs::create_dir_all(&dist).unwrap();
        fs::write(dist.join("bundle.js"), "var x=1").unwrap();

        let build = dir.path().join("build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("output.js"), "compiled").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(paths.contains(&"app.py"));
        assert!(
            !paths.iter().any(|p| p.starts_with(".venv")),
            ".venv should be excluded"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("dist")),
            "dist should be excluded"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("build")),
            "build should be excluded"
        );
    }

    #[test]
    fn gitignore_patterns_with_wildcards() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("test.log"), "log data").unwrap();
        fs::write(dir.path().join("app.log"), "more logs").unwrap();
        fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(!names.contains(&"test.log"));
        assert!(!names.contains(&"app.log"));
    }

    #[test]
    fn handles_invalid_path() {
        let result = discover_files(Path::new("/nonexistent/path"), &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn veraignore_overrides_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join("notes.md"), "local notes").unwrap();
        // .gitignore excludes secret.txt and notes.md
        fs::write(dir.path().join(".gitignore"), "secret.txt\nnotes.md\n").unwrap();
        // .veraignore only excludes secret.txt (notes.md should now be indexed)
        fs::write(dir.path().join(".veraignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.contains(&"notes.md"),
            "notes.md should be indexed when .veraignore overrides .gitignore"
        );
        assert!(
            !names.contains(&"secret.txt"),
            "secret.txt should still be excluded by .veraignore"
        );
    }

    #[test]
    fn veraignore_include_gitignore_directive() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join("draft.md"), "draft notes").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();
        // .veraignore includes gitignore rules AND adds draft.md exclusion
        fs::write(
            dir.path().join(".veraignore"),
            "#include .gitignore\ndraft.md\n",
        )
        .unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignore rules should apply via #include"
        );
        assert!(
            !names.contains(&"draft.md"),
            ".veraignore extra pattern should apply"
        );
    }

    #[test]
    fn no_veraignore_uses_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignore should apply when no .veraignore"
        );
    }

    #[test]
    fn extra_excludes_from_cli() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let vendor = dir.path().join("vendor");
        fs::create_dir_all(&vendor).unwrap();
        fs::write(vendor.join("dep.rs"), "fn dep() {}").unwrap();

        let mut config = default_config();
        config.extra_excludes = vec!["vendor/**".to_string()];

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.iter().any(|n| n.starts_with("vendor")),
            "--exclude should exclude vendor"
        );
    }

    #[test]
    fn no_ignore_disables_gitignore_and_veraignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let mut config = default_config();
        config.no_ignore = true;

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.contains(&"secret.txt"),
            "no_ignore should disable gitignore"
        );
    }

    #[test]
    fn no_default_excludes_disables_smart_defaults() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let nm = dir.path().join("node_modules");
        fs::create_dir_all(&nm).unwrap();
        fs::write(nm.join("dep.js"), "module.exports = {}").unwrap();

        let mut config = default_config();
        config.no_default_excludes = true;

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.iter().any(|n| n.starts_with("node_modules")),
            "no_default_excludes should allow node_modules"
        );
    }

    #[test]
    fn veraignore_itself_excluded_from_indexing() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join(".veraignore"), "*.log\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&".veraignore"),
            ".veraignore should not be indexed"
        );
    }

    #[test]
    fn default_excludes_can_exclude_exact_filename() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("UPGRADE.md"), "notes").unwrap();
        fs::create_dir_all(dir.path().join("UPGRADE.md-dir")).unwrap();
        fs::write(
            dir.path().join("UPGRADE.md-dir").join("keep.rs"),
            "fn keep() {}",
        )
        .unwrap();

        let mut config = default_config();
        config.default_excludes = vec!["UPGRADE.md".to_string()];

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(!names.contains(&"UPGRADE.md"));
        assert!(
            names.iter().any(|p| p.starts_with("UPGRADE.md-dir/")),
            "only exact filename should be excluded, not similarly named directory"
        );
    }
}
