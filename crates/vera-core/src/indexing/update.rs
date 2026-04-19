//! Incremental index update logic.
//!
//! Detects changed files via content hashing, then re-indexes only
//! modified/new files and removes deleted files from the index.
//!
//! The algorithm:
//! 1. Discover current files on disk
//! 2. Load stored content hashes from the metadata DB
//! 3. Classify each file as: unchanged, modified, new, or deleted
//! 4. For modified/new files: re-parse, re-chunk, re-embed, update stores
//! 5. For deleted files: remove chunks, vectors, BM25 entries, and hashes
//! 6. Return an UpdateSummary describing what changed

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::config::VeraConfig;
use crate::discovery;
use crate::embedding::{EmbeddingProvider, embed_chunks_concurrent};
use crate::parsing;
use crate::storage::bm25::{Bm25Document, Bm25Index};
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;
use crate::types::Language;

use super::pipeline;

/// Summary of an incremental update run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UpdateSummary {
    /// Files that were modified and re-indexed.
    pub files_modified: usize,
    /// New files that were indexed.
    pub files_added: usize,
    /// Files that were deleted from the index.
    pub files_deleted: usize,
    /// Files that were unchanged (skipped).
    pub files_unchanged: usize,
    /// Total chunks after the update.
    pub total_chunks: u64,
    /// Wall-clock elapsed time in seconds.
    pub elapsed_secs: f64,
}

/// Compute a SHA-256 content hash for a file's contents.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    hash.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn detect_language_for_path(file_path: &str) -> Language {
    Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(Language::from_filename)
        .unwrap_or_else(|| {
            let ext = Path::new(file_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            Language::from_extension(ext)
        })
}

fn hash_for_indexing_source(
    content: &str,
    rel_path: &str,
    language: Language,
    repo_root: &Path,
) -> String {
    if language != Language::Rst {
        return content_hash(content);
    }

    let absolute_path = repo_root.join(rel_path);
    match parsing::sphinx::preprocess_rst(content, &absolute_path, repo_root) {
        Ok(preprocessed) => content_hash(&preprocessed),
        Err(err) => {
            warn!(
                file = %rel_path,
                error = %err,
                "failed to preprocess rst for hashing; falling back to raw source"
            );
            content_hash(content)
        }
    }
}

/// Incrementally update the index for a repository.
///
/// Only re-indexes files whose content has changed since the last index/update.
/// Handles:
/// - Modified files: re-parse, re-chunk, re-embed, update all stores
/// - New files: parse, chunk, embed, add to all stores
/// - Deleted files: remove from all stores
/// - Unchanged files: skip entirely
pub async fn update_repository<P: EmbeddingProvider>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
) -> Result<UpdateSummary> {
    let start = Instant::now();

    // ── 1. Validate path ─────────────────────────────────────────
    if !repo_path.exists() {
        bail!("path does not exist: {}", repo_path.display());
    }
    if !repo_path.is_dir() {
        bail!("path is not a directory: {}", repo_path.display());
    }

    let repo_root = repo_path
        .canonicalize()
        .with_context(|| format!("failed to resolve path: {}", repo_path.display()))?;

    let idx_dir = pipeline::index_dir(&repo_root);
    if !idx_dir.exists() {
        bail!(
            "no index found at {}. Run `vera index` first.",
            idx_dir.display()
        );
    }

    info!(path = %repo_root.display(), "starting incremental update");

    // ── 2. Discover current files on disk ────────────────────────
    let disc =
        discovery::discover_files(&repo_root, &config.indexing).context("file discovery failed")?;

    // ── 3. Load stored hashes and classify files ─────────────────
    let metadata_path = idx_dir.join("metadata.db");
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;

    let mut stored_dim = config.embedding.max_stored_dim;

    // Check for provider mismatch.
    if let (Some(s_model), Some(s_dim)) = (
        metadata_store.get_index_meta("model_name").unwrap_or(None),
        metadata_store
            .get_index_meta("embedding_dim")
            .unwrap_or(None),
    ) {
        if !crate::config::model_names_match(&s_model, model_name) {
            bail!(
                "Index was created with model '{}' ({} dimensions), but you are using model '{}'. Please re-index with matching provider.",
                s_model,
                s_dim,
                model_name
            );
        }
        if let Ok(dim) = s_dim.parse::<usize>() {
            if let Some(provider_dim) = provider.expected_dim() {
                if provider_dim != dim {
                    bail!(
                        "Dimension mismatch: index has {} dimensions but active provider expects {}. Please re-index with matching provider.",
                        dim,
                        provider_dim
                    );
                }
            }
            stored_dim = dim;
        }
    } else if let Some(s_dim) = metadata_store
        .get_index_meta("embedding_dim")
        .unwrap_or(None)
    {
        if let Ok(dim) = s_dim.parse::<usize>() {
            stored_dim = dim;
        }
    }

    let stored_files: HashSet<String> = metadata_store
        .indexed_files()
        .context("failed to list indexed files")?
        .into_iter()
        .collect();

    // Read file contents and compute hashes for current files.
    let mut current_files: HashMap<String, String> = HashMap::new(); // rel_path → content
    for file in &disc.files {
        match std::fs::read_to_string(&file.absolute_path) {
            Ok(content) => {
                current_files.insert(file.relative_path.clone(), content);
            }
            Err(err) => {
                warn!(file = %file.relative_path, error = %err, "failed to read file");
            }
        }
    }

    let current_paths: HashSet<&str> = current_files.keys().map(|s| s.as_str()).collect();

    // Classify files.
    let mut modified = Vec::new();
    let mut added = Vec::new();
    let mut deleted = Vec::new();
    let mut unchanged = 0usize;

    for (rel_path, content) in &current_files {
        let language = detect_language_for_path(rel_path);
        let hash = hash_for_indexing_source(content, rel_path, language, &repo_root);
        let stored_hash = metadata_store
            .get_file_hash(rel_path)
            .context("failed to get stored hash")?;

        if stored_files.contains(rel_path.as_str()) {
            // File exists in index.
            match stored_hash {
                Some(ref old_hash) if *old_hash == hash => {
                    unchanged += 1;
                }
                _ => {
                    modified.push((rel_path.clone(), content.clone(), hash));
                }
            }
        } else {
            // New file (not in index).
            added.push((rel_path.clone(), content.clone(), hash));
        }
    }

    for stored_path in &stored_files {
        if !current_paths.contains(stored_path.as_str()) {
            deleted.push(stored_path.clone());
        }
    }

    info!(
        modified = modified.len(),
        added = added.len(),
        deleted = deleted.len(),
        unchanged,
        "file classification complete"
    );

    // ── 4. Process deletions ─────────────────────────────────────
    if !deleted.is_empty() {
        let vector_path = idx_dir.join("vectors.db");
        let vector_store = VectorStore::open(&vector_path, stored_dim)
            .context("failed to open vector store for deletion")?;
        let bm25_dir = idx_dir.join("bm25");
        let bm25_index =
            Bm25Index::open(&bm25_dir).context("failed to open BM25 index for deletion")?;

        for file_path in &deleted {
            remove_file_from_index(&metadata_store, &vector_store, &bm25_index, file_path)?;
        }
    }

    // ── 5. Process modifications and additions ───────────────────
    let files_to_index: Vec<(String, String, String)> =
        modified.iter().chain(added.iter()).cloned().collect();

    if !files_to_index.is_empty() {
        // For modified files, first remove old data.
        if !modified.is_empty() {
            let vector_path = idx_dir.join("vectors.db");
            let vector_store = VectorStore::open(&vector_path, stored_dim)
                .context("failed to open vector store for modification")?;
            let bm25_dir = idx_dir.join("bm25");
            let bm25_index =
                Bm25Index::open(&bm25_dir).context("failed to open BM25 index for modification")?;

            for (file_path, _, _) in &modified {
                remove_file_from_index(&metadata_store, &vector_store, &bm25_index, file_path)?;
            }
        }

        // Parse and chunk new/modified files.
        let mut all_chunks = Vec::new();
        for (rel_path, content, _hash) in &files_to_index {
            let language = detect_language_for_path(rel_path);

            // For RST, refs come from raw source; chunks from preprocessed.
            // For all other languages, parse once for both.
            let (chunks, refs) = if language == Language::Rst {
                let refs = parsing::parse_and_extract_references(content, language);
                let absolute_path = repo_root.join(rel_path);
                let normalized_source =
                    match parsing::sphinx::preprocess_rst(content, &absolute_path, &repo_root) {
                        Ok(preprocessed) => Some(preprocessed),
                        Err(err) => {
                            warn!(
                                file = %rel_path,
                                error = %err,
                                "failed to preprocess rst during update; falling back to raw source"
                            );
                            None
                        }
                    };
                let src = normalized_source.as_deref().unwrap_or(content);
                match parsing::parse_and_chunk(src, rel_path, language, &config.indexing) {
                    Ok(chunks) => (chunks, refs),
                    Err(err) => {
                        warn!(
                            file = %rel_path,
                            error = %err,
                            refs = refs.len(),
                            "failed to chunk rst during update; keeping extracted references"
                        );
                        (Vec::new(), refs)
                    }
                }
            } else {
                match parsing::parse_file(content, rel_path, language, &config.indexing) {
                    Ok(result) => result,
                    Err(err) => {
                        warn!(file = %rel_path, error = %err, "parse error during update");
                        continue;
                    }
                }
            };

            if !refs.is_empty() {
                metadata_store
                    .insert_references(rel_path, &refs)
                    .context("failed to store references")?;
            }

            debug!(file = %rel_path, chunks = chunks.len(), refs = refs.len(), "parsed file");
            all_chunks.extend(chunks);
        }

        if !all_chunks.is_empty() {
            // Generate embeddings.
            let mut embeddings = embed_chunks_concurrent(
                provider,
                &all_chunks,
                config.embedding.batch_size,
                config.embedding.max_concurrent_requests,
                config.indexing.max_chunk_bytes,
            )
            .await
            .context("embedding generation failed")?;

            // Truncate if needed.
            let final_stored_dim = super::truncate_embeddings(&mut embeddings, stored_dim);

            // Store metadata.
            metadata_store
                .insert_chunks(&all_chunks)
                .context("failed to insert updated chunk metadata")?;

            // Store vectors.
            let vector_path = idx_dir.join("vectors.db");
            let vector_store = VectorStore::open(&vector_path, final_stored_dim)
                .context("failed to open vector store for insertion")?;

            let batch: Vec<(&str, &[f32])> = embeddings
                .iter()
                .map(|(id, vec)| (id.as_str(), vec.as_slice()))
                .collect();
            vector_store
                .insert_batch(&batch)
                .context("failed to insert updated vectors")?;

            // Store BM25 documents.
            let bm25_dir = idx_dir.join("bm25");
            let bm25_index =
                Bm25Index::open(&bm25_dir).context("failed to open BM25 index for insertion")?;

            let lang_strings: Vec<String> =
                all_chunks.iter().map(|c| c.language.to_string()).collect();
            let bm25_docs: Vec<Bm25Document<'_>> = all_chunks
                .iter()
                .zip(lang_strings.iter())
                .map(|(c, lang)| Bm25Document {
                    chunk_id: &c.id,
                    file_path: &c.file_path,
                    content: &c.content,
                    symbol_name: c.symbol_name.as_deref(),
                    language: lang,
                })
                .collect();
            bm25_index
                .insert_batch(&bm25_docs)
                .context("failed to insert updated BM25 documents")?;
        }

        // Update file hashes for all indexed files.
        for (rel_path, _content, hash) in &files_to_index {
            metadata_store
                .set_file_hash(rel_path, hash)
                .context("failed to update file hash")?;
        }
    }

    // ── 6. Get final counts ──────────────────────────────────────
    let total_chunks = metadata_store
        .chunk_count()
        .context("failed to count chunks")?;

    let summary = UpdateSummary {
        files_modified: modified.len(),
        files_added: added.len(),
        files_deleted: deleted.len(),
        files_unchanged: unchanged,
        total_chunks,
        elapsed_secs: start.elapsed().as_secs_f64(),
    };

    info!(
        modified = summary.files_modified,
        added = summary.files_added,
        deleted = summary.files_deleted,
        unchanged = summary.files_unchanged,
        total_chunks = summary.total_chunks,
        elapsed = %format!("{:.2}s", summary.elapsed_secs),
        "incremental update complete"
    );

    Ok(summary)
}

/// Remove all data for a file from the index stores.
fn remove_file_from_index(
    metadata_store: &MetadataStore,
    vector_store: &VectorStore,
    bm25_index: &Bm25Index,
    file_path: &str,
) -> Result<()> {
    metadata_store
        .delete_references_by_file(file_path)
        .context("failed to delete references for file")?;
    // Get chunk IDs for this file (needed for vector/BM25 deletion).
    let chunks = metadata_store
        .get_chunks_by_file(file_path)
        .context("failed to get chunks for file deletion")?;

    // Delete from vector store using file prefix pattern.
    let prefix = format!("{file_path}:");
    vector_store
        .delete_by_file_prefix(&prefix)
        .with_context(|| format!("failed to delete vectors for {file_path}"))?;

    // Delete from BM25 index by file path.
    bm25_index
        .delete_by_file(file_path)
        .with_context(|| format!("failed to delete BM25 entries for {file_path}"))?;

    // Delete chunk metadata.
    metadata_store
        .delete_chunks_by_file(file_path)
        .with_context(|| format!("failed to delete metadata for {file_path}"))?;

    // Delete file hash.
    metadata_store
        .delete_file_hash(file_path)
        .with_context(|| format!("failed to delete file hash for {file_path}"))?;

    debug!(
        file = %file_path,
        chunks = chunks.len(),
        "removed file from index"
    );

    Ok(())
}

#[cfg(test)]
#[path = "update_tests.rs"]
mod tests;
