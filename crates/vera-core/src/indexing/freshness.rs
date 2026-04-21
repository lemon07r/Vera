//! Index freshness metadata and stale-index detection.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tracing::warn;

use crate::config::IndexingConfig;
use crate::discovery;
use crate::storage::metadata::MetadataStore;

use super::index_dir;
use super::update::{detect_language_for_path, hash_for_indexing_source};

const INDEXING_CONFIG_KEY: &str = "indexing_config";
const INDEX_REFRESHED_AT_KEY: &str = "index_refreshed_at_unix_ms";

/// Summary of drift between the working tree and the current index.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct IndexFreshness {
    pub files_added: usize,
    pub files_modified: usize,
    pub files_deleted: usize,
}

impl IndexFreshness {
    pub fn is_stale(&self) -> bool {
        self.total_changes() > 0
    }

    pub fn total_changes(&self) -> usize {
        self.files_added + self.files_modified + self.files_deleted
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.files_added > 0 {
            parts.push(format!("{} added", self.files_added));
        }
        if self.files_modified > 0 {
            parts.push(format!("{} modified", self.files_modified));
        }
        if self.files_deleted > 0 {
            parts.push(format!("{} deleted", self.files_deleted));
        }
        parts.join(", ")
    }
}

pub(crate) fn record_index_snapshot(
    metadata_store: &MetadataStore,
    indexing_config: &IndexingConfig,
) -> Result<()> {
    metadata_store
        .set_index_meta(
            INDEXING_CONFIG_KEY,
            &serde_json::to_string(indexing_config).context("failed to encode indexing config")?,
        )
        .context("failed to store indexing config metadata")?;
    metadata_store
        .set_index_meta(
            INDEX_REFRESHED_AT_KEY,
            &current_time_millis()
                .context("failed to compute index refresh timestamp")?
                .to_string(),
        )
        .context("failed to store index refresh timestamp")?;
    Ok(())
}

/// Compare the current repo contents against the index metadata.
///
/// New and deleted files are detected via discovery vs tracked files. Modified
/// files are verified with the stored content hashes, using the saved refresh
/// timestamp to avoid re-hashing unchanged candidates when available.
pub fn detect_staleness(
    repo_path: &Path,
    fallback_config: &IndexingConfig,
) -> Result<IndexFreshness> {
    let repo_root = repo_path
        .canonicalize()
        .with_context(|| format!("failed to resolve repo path: {}", repo_path.display()))?;
    let metadata_path = index_dir(&repo_root).join("metadata.db");
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;

    let indexing_config = load_indexing_config(&metadata_store, fallback_config);
    let refreshed_at_ms = load_refreshed_at_millis(&metadata_store);
    let discovery = discovery::discover_files(&repo_root, &indexing_config)
        .context("failed to discover files for freshness scan")?;

    let current_files: HashMap<String, PathBuf> = discovery
        .files
        .into_iter()
        .map(|file| (file.relative_path, file.absolute_path))
        .collect();
    let tracked_files: HashSet<String> = metadata_store
        .tracked_files()
        .context("failed to read tracked files")?
        .into_iter()
        .collect();

    let files_added = current_files
        .keys()
        .filter(|path| !tracked_files.contains(path.as_str()))
        .count();
    let files_deleted = tracked_files
        .iter()
        .filter(|path| !current_files.contains_key(path.as_str()))
        .count();

    let mut files_modified = 0usize;
    for (rel_path, absolute_path) in current_files
        .iter()
        .filter(|(path, _)| tracked_files.contains(path.as_str()))
        .filter(|(_, absolute_path)| {
            refreshed_at_ms
                .is_none_or(|refresh_time| file_may_be_newer(absolute_path, refresh_time))
        })
    {
        let content = match std::fs::read_to_string(absolute_path) {
            Ok(content) => content,
            Err(err) => {
                warn!(
                    file = %rel_path,
                    error = %err,
                    "failed to read file during freshness scan"
                );
                continue;
            }
        };
        let language = detect_language_for_path(rel_path);
        let current_hash = hash_for_indexing_source(&content, rel_path, language, &repo_root);
        let stored_hash = metadata_store
            .get_file_hash(rel_path)
            .with_context(|| format!("failed to read stored hash for {rel_path}"))?;
        if stored_hash.as_deref() != Some(current_hash.as_str()) {
            files_modified += 1;
        }
    }

    Ok(IndexFreshness {
        files_added,
        files_modified,
        files_deleted,
    })
}

fn load_indexing_config(
    metadata_store: &MetadataStore,
    fallback_config: &IndexingConfig,
) -> IndexingConfig {
    match metadata_store.get_index_meta(INDEXING_CONFIG_KEY) {
        Ok(Some(encoded)) => match serde_json::from_str(&encoded) {
            Ok(config) => config,
            Err(err) => {
                warn!(error = %err, "failed to decode saved indexing config");
                fallback_config.clone()
            }
        },
        Ok(None) => fallback_config.clone(),
        Err(err) => {
            warn!(error = %err, "failed to read saved indexing config");
            fallback_config.clone()
        }
    }
}

fn load_refreshed_at_millis(metadata_store: &MetadataStore) -> Option<u64> {
    match metadata_store.get_index_meta(INDEX_REFRESHED_AT_KEY) {
        Ok(Some(raw)) => match raw.parse::<u64>() {
            Ok(value) => Some(value),
            Err(err) => {
                warn!(value = %raw, error = %err, "failed to parse saved refresh timestamp");
                None
            }
        },
        Ok(None) => None,
        Err(err) => {
            warn!(error = %err, "failed to read saved refresh timestamp");
            None
        }
    }
}

fn file_may_be_newer(path: &Path, refreshed_at_ms: u64) -> bool {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64 > refreshed_at_ms)
        .unwrap_or(true)
}

fn current_time_millis() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexingConfig;
    use crate::indexing::content_hash;
    use tempfile::tempdir;

    fn write_file(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn detects_added_modified_and_deleted_files() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", "pub fn current() {}\n");
        write_file(dir.path(), "src/new.rs", "pub fn added() {}\n");

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        metadata
            .set_file_hash("src/lib.rs", &content_hash("pub fn previous() {}\n"))
            .unwrap();
        metadata
            .set_file_hash("src/deleted.rs", &content_hash("pub fn deleted() {}\n"))
            .unwrap();

        let freshness = detect_staleness(dir.path(), &IndexingConfig::default()).unwrap();
        assert_eq!(
            freshness,
            IndexFreshness {
                files_added: 1,
                files_modified: 1,
                files_deleted: 1,
            }
        );
    }

    #[test]
    fn freshness_scan_uses_saved_indexing_config() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "generated/out.rs", "pub fn generated() {}\n");

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();

        let mut saved_config = IndexingConfig::default();
        saved_config.extra_excludes = vec!["generated/**".to_string()];
        record_index_snapshot(&metadata, &saved_config).unwrap();

        let freshness = detect_staleness(dir.path(), &IndexingConfig::default()).unwrap();
        assert_eq!(freshness.files_added, 0);
    }
}
