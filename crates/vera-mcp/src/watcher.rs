//! Background file watcher for automatic index updates in MCP mode.
//!
//! Watches a project directory for file changes and triggers incremental
//! index updates after a debounce period. This keeps the index fresh
//! without requiring manual update calls.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use notify_debouncer_mini::{DebouncedEventKind, new_debouncer};
use tracing::{debug, info, warn};

/// Debounce interval: wait this long after the last file change before updating.
const DEBOUNCE_SECS: u64 = 2;

/// Handle to a running file watcher. Dropping it stops the watcher.
pub struct WatchHandle {
    _watcher: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
    /// Set to true when an update is in progress.
    updating: Arc<AtomicBool>,
}

impl WatchHandle {
    /// True if an incremental update is currently running.
    pub fn is_updating(&self) -> bool {
        self.updating.load(Ordering::Relaxed)
    }
}

/// Start watching a project directory for file changes.
///
/// When changes are detected (after debouncing), triggers an incremental
/// index update in a background thread. Returns a handle that keeps the
/// watcher alive; drop it to stop watching.
pub fn start_watching(repo_path: &Path) -> Result<WatchHandle, String> {
    let repo_path = repo_path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve path: {e}"))?;

    let idx_dir = vera_core::indexing::index_dir(&repo_path);
    if !idx_dir.exists() {
        return Err("No index found. Run search_code first to auto-index.".to_string());
    }

    let updating = Arc::new(AtomicBool::new(false));
    let updating_clone = updating.clone();
    let repo_clone = repo_path.clone();

    let mut debouncer = new_debouncer(
        Duration::from_secs(DEBOUNCE_SECS),
        move |events: Result<Vec<notify_debouncer_mini::DebouncedEvent>, notify::Error>| {
            let events = match events {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "File watcher error");
                    return;
                }
            };

            // Filter out events inside .vera/ directory.
            let has_relevant_changes = events.iter().any(|e| {
                e.kind == DebouncedEventKind::Any && !e.path.starts_with(repo_clone.join(".vera"))
            });

            if !has_relevant_changes {
                return;
            }

            // Skip if already updating.
            if updating_clone
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_err()
            {
                debug!("Skipping auto-update: previous update still running");
                return;
            }

            let repo = repo_clone.clone();
            let flag = updating_clone.clone();

            std::thread::spawn(move || {
                run_incremental_update(&repo, &flag);
            });
        },
    )
    .map_err(|e| format!("Failed to create file watcher: {e}"))?;

    debouncer
        .watcher()
        .watch(&repo_path, notify::RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to watch directory: {e}"))?;

    info!(path = %repo_path.display(), "Started file watcher for auto-indexing");

    Ok(WatchHandle {
        _watcher: debouncer,
        updating,
    })
}

/// Run an incremental update, resetting the flag when done.
fn run_incremental_update(repo_path: &Path, updating: &AtomicBool) {
    debug!(path = %repo_path.display(), "Auto-update triggered by file changes");

    let result = run_update_blocking(repo_path);

    match result {
        Ok(summary) => {
            let changed = summary.files_modified + summary.files_added + summary.files_deleted;
            if changed > 0 {
                info!(
                    modified = summary.files_modified,
                    added = summary.files_added,
                    deleted = summary.files_deleted,
                    "Auto-update complete"
                );
            } else {
                debug!("Auto-update: no changes detected");
            }
        }
        Err(e) => {
            warn!(error = %e, "Auto-update failed");
        }
    }

    updating.store(false, Ordering::SeqCst);
}

/// Blocking wrapper around the async update_repository.
fn run_update_blocking(
    repo_path: &Path,
) -> Result<vera_core::indexing::UpdateSummary, anyhow::Error> {
    let backend = vera_core::config::resolve_backend(None);
    let mut config = vera_core::config::VeraConfig::default();
    config.adjust_for_backend(backend);

    let rt = tokio::runtime::Runtime::new()?;

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    rt.block_on(vera_core::indexing::update_repository(
        repo_path,
        &provider,
        &config,
        &model_name,
    ))
}
