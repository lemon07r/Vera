//! Indexing pipeline orchestrator.
//!
//! Coordinates file discovery, parsing, chunking, embedding, and storage
//! into a single `index_repository` entry point. Produces an [`IndexSummary`]
//! describing the work performed.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::config::VeraConfig;
use crate::discovery::{self, DiscoveryResult};
use crate::embedding::{EmbeddingProvider, embed_chunks_concurrent_with_progress};
use crate::indexing::update::content_hash;
use crate::parsing;
use crate::parsing::references::RawReference;
use crate::storage::bm25::{Bm25Document, Bm25Index};
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;
use crate::types::{Chunk, Language};

// ── Index summary ────────────────────────────────────────────────────

/// Summary of an indexing run, suitable for display to the user.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexSummary {
    /// Number of source files parsed.
    pub files_parsed: usize,
    /// Number of chunks created from parsed files.
    pub chunks_created: usize,
    /// Number of embedding vectors generated.
    pub embeddings_generated: usize,
    /// Number of binary files skipped.
    pub binary_skipped: usize,
    /// Number of files skipped due to size threshold.
    pub large_skipped: usize,
    /// Relative paths and sizes (bytes) of files skipped due to size threshold.
    pub large_skipped_paths: Vec<(String, u64)>,
    /// Number of files skipped due to permission or read errors.
    pub error_skipped: usize,
    /// Files that had parse errors (path + error message).
    pub parse_errors: Vec<FileError>,
    /// Wall-clock elapsed time in seconds.
    pub elapsed_secs: f64,
}

/// A file-level error encountered during indexing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FileError {
    pub file_path: String,
    pub error: String,
}

// ── Progress reporting ───────────────────────────────────────────────

/// Progress events emitted during indexing.
#[derive(Debug, Clone)]
pub enum IndexProgress {
    /// File discovery complete.
    DiscoveryDone { file_count: usize },
    /// Parsing and chunking complete.
    ParsingDone { chunk_count: usize },
    /// An embedding batch finished. `done` is cumulative chunks embedded so far.
    EmbeddingProgress { done: usize, total: usize },
    /// All embeddings generated.
    EmbeddingDone { count: usize },
    /// Index artifacts written to disk.
    StorageDone,
}

/// No-op progress callback (used when caller doesn't need progress).
pub(crate) fn no_progress(_: IndexProgress) {}

// ── Index directory layout ───────────────────────────────────────────

/// Default index directory name (placed inside the indexed repo).
const INDEX_DIR_NAME: &str = ".vera";

/// Subdirectory for BM25 (Tantivy) index files.
const BM25_SUBDIR: &str = "bm25";

/// Filename for SQLite metadata + vector databases.
const METADATA_DB: &str = "metadata.db";
const VECTOR_DB: &str = "vectors.db";

/// Resolve the index directory for a given repository root.
pub fn index_dir(repo_root: &Path) -> std::path::PathBuf {
    repo_root.join(INDEX_DIR_NAME)
}

// ── Pipeline entry point ─────────────────────────────────────────────

/// Index a repository: discover files, parse, chunk, embed, and store.
///
/// This is the main orchestrator for `vera index <path>`. It:
/// 1. Validates the input path
/// 2. Discovers source files (respecting .gitignore and exclusions)
/// 3. Parses and chunks each file
/// 4. Generates embeddings via the provider
/// 5. Stores metadata, vectors, and BM25 index on disk
///
/// # Arguments
/// - `repo_path` — Path to the repository to index
/// - `provider` — Embedding provider (API-backed or mock)
/// - `config` — Pipeline configuration
///
/// # Errors
/// Returns an error if the path is invalid, not a directory, or storage fails.
pub async fn index_repository<P: EmbeddingProvider>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
) -> Result<IndexSummary> {
    index_repository_with_progress(repo_path, provider, config, model_name, no_progress).await
}

/// Index a repository with progress reporting via a callback.
pub async fn index_repository_with_progress<P, F>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
    on_progress: F,
) -> Result<IndexSummary>
where
    P: EmbeddingProvider,
    F: Fn(IndexProgress) + Send + Sync,
{
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

    info!(path = %repo_root.display(), "starting indexing");

    // ── 2. Discover files ────────────────────────────────────────
    let discovery =
        discovery::discover_files(&repo_root, &config.indexing).context("file discovery failed")?;

    if discovery.files.is_empty() {
        return Ok(IndexSummary {
            files_parsed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            binary_skipped: discovery.binary_skipped,
            large_skipped: discovery.large_skipped,
            large_skipped_paths: discovery.large_skipped_paths.clone(),
            error_skipped: discovery.error_skipped,
            parse_errors: Vec::new(),
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }

    info!(
        files = discovery.files.len(),
        binary_skipped = discovery.binary_skipped,
        large_skipped = discovery.large_skipped,
        error_skipped = discovery.error_skipped,
        "file discovery complete"
    );
    on_progress(IndexProgress::DiscoveryDone {
        file_count: discovery.files.len(),
    });

    // ── 3. Parse and chunk each file (parallelized with rayon) ──
    let (all_chunks, parse_errors, file_hashes, all_refs) =
        parse_discovered_files_parallel(&discovery, &repo_root, config);

    info!(
        chunks = all_chunks.len(),
        parse_errors = parse_errors.len(),
        "parsing complete"
    );
    on_progress(IndexProgress::ParsingDone {
        chunk_count: all_chunks.len(),
    });

    if all_chunks.is_empty() {
        return Ok(IndexSummary {
            files_parsed: discovery.files.len() - parse_errors.len(),
            chunks_created: 0,
            embeddings_generated: 0,
            binary_skipped: discovery.binary_skipped,
            large_skipped: discovery.large_skipped,
            large_skipped_paths: discovery.large_skipped_paths.clone(),
            error_skipped: discovery.error_skipped,
            parse_errors,
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }

    // ── 4. Generate embeddings (concurrent batches) ──────────────
    let batch_size = config.embedding.batch_size;
    let max_concurrent_requests = config.embedding.max_concurrent_requests;

    let progress_cb = |done: usize, total: usize| {
        on_progress(IndexProgress::EmbeddingProgress { done, total });
    };
    let mut embeddings = embed_chunks_concurrent_with_progress(
        provider,
        &all_chunks,
        batch_size,
        max_concurrent_requests,
        config.indexing.max_chunk_bytes,
        progress_cb,
    )
    .await
    .context("embedding generation failed")?;

    // Truncate vectors if max_stored_dim is configured.
    let stored_dim = super::truncate_embeddings(&mut embeddings, config.embedding.max_stored_dim);

    info!(
        embeddings = embeddings.len(),
        stored_dim, "embeddings generated"
    );
    on_progress(IndexProgress::EmbeddingDone {
        count: embeddings.len(),
    });

    // ── 5. Store everything on disk ──────────────────────────────
    let idx_dir = index_dir(&repo_root);
    store_index(
        &idx_dir,
        &all_chunks,
        &embeddings,
        &file_hashes,
        &all_refs,
        model_name,
    )
    .context("failed to write index artifacts")?;

    info!(index_dir = %idx_dir.display(), "index artifacts written");
    on_progress(IndexProgress::StorageDone);

    let files_parsed = discovery.files.len() - parse_errors.len();

    Ok(IndexSummary {
        files_parsed,
        chunks_created: all_chunks.len(),
        embeddings_generated: embeddings.len(),
        binary_skipped: discovery.binary_skipped,
        large_skipped: discovery.large_skipped,
        large_skipped_paths: discovery.large_skipped_paths,
        error_skipped: discovery.error_skipped,
        parse_errors,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Parse all discovered files in parallel using rayon and collect chunks.
///
/// Each file is read and parsed on a rayon thread pool worker. Results
/// are collected and flattened. Files that fail parsing are recorded as
/// errors but do not abort the pipeline. Also computes content hashes
/// for incremental indexing support.
#[allow(clippy::type_complexity)]
fn parse_discovered_files_parallel(
    discovery: &DiscoveryResult,
    repo_root: &Path,
    config: &VeraConfig,
) -> (
    Vec<Chunk>,
    Vec<FileError>,
    Vec<(String, String)>,
    Vec<(String, Vec<RawReference>)>,
) {
    let config = Arc::new(config.clone());
    let repo_root = Arc::new(repo_root.to_path_buf());

    // Process files in parallel: returns Ok((chunks, rel_path, hash, refs)) or Err.
    #[allow(clippy::type_complexity)]
    let results: Vec<Result<(Vec<Chunk>, String, String, Vec<RawReference>), FileError>> =
        discovery
            .files
            .par_iter()
            .map(|file| {
                let source = std::fs::read_to_string(&file.absolute_path).map_err(|err| {
                    warn!(
                        file = %file.relative_path,
                        error = %err,
                        "failed to read file for parsing"
                    );
                    FileError {
                        file_path: file.relative_path.clone(),
                        error: err.to_string(),
                    }
                })?;

                let language = file
                    .absolute_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(Language::from_filename)
                    .unwrap_or_else(|| {
                        let ext = file
                            .absolute_path
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("");
                        Language::from_extension(ext)
                    });

                // RST files need preprocessing before chunking, but refs
                // come from the raw source, so they can't share a single parse.
                let parse_result = if language == Language::Rst {
                    let refs = parsing::parse_and_extract_references(&source, language);
                    let normalized_source = match parsing::sphinx::preprocess_rst(
                        &source,
                        &file.absolute_path,
                        repo_root.as_path(),
                    ) {
                        Ok(preprocessed) => Some(preprocessed),
                        Err(err) => {
                            warn!(
                                file = %file.relative_path,
                                error = %err,
                                "failed to preprocess rst; falling back to raw source"
                            );
                            None
                        }
                    };
                    let src = normalized_source.as_deref().unwrap_or(&source);
                    let hash = content_hash(src);
                    parsing::parse_and_chunk(src, &file.relative_path, language, &config.indexing)
                        .map(|chunks| (chunks, refs, hash))
                } else {
                    let hash = content_hash(&source);
                    parsing::parse_file(&source, &file.relative_path, language, &config.indexing)
                        .map(|(chunks, refs)| (chunks, refs, hash))
                };

                parse_result
                    .inspect(|(chunks, refs, _)| {
                        debug!(
                            file = %file.relative_path,
                            chunks = chunks.len(),
                            refs = refs.len(),
                            "parsed file"
                        );
                    })
                    .map(|(chunks, refs, hash)| (chunks, file.relative_path.clone(), hash, refs))
                    .map_err(|err| {
                        warn!(
                            file = %file.relative_path,
                            error = %err,
                            "parse error"
                        );
                        FileError {
                            file_path: file.relative_path.clone(),
                            error: err.to_string(),
                        }
                    })
            })
            .collect();

    // Flatten results into chunks, errors, file hashes, and references.
    let mut all_chunks = Vec::new();
    let mut parse_errors = Vec::new();
    let mut file_hashes = Vec::new();
    let mut all_refs = Vec::new();
    for result in results {
        match result {
            Ok((chunks, rel_path, hash, refs)) => {
                all_chunks.extend(chunks);
                if !refs.is_empty() {
                    all_refs.push((rel_path.clone(), refs));
                }
                file_hashes.push((rel_path, hash));
            }
            Err(error) => parse_errors.push(error),
        }
    }

    (all_chunks, parse_errors, file_hashes, all_refs)
}

/// Write chunks, embeddings, BM25 index, file hashes, and references to disk.
fn store_index(
    idx_dir: &Path,
    chunks: &[Chunk],
    embeddings: &[(String, Vec<f32>)],
    file_hashes: &[(String, String)],
    file_refs: &[(String, Vec<RawReference>)],
    model_name: &str,
) -> Result<()> {
    // Ensure index directory exists.
    std::fs::create_dir_all(idx_dir)
        .with_context(|| format!("failed to create index dir: {}", idx_dir.display()))?;

    // Determine vector dimensionality from the first embedding.
    let dim = embeddings.first().map(|(_, v)| v.len()).unwrap_or(4096);

    // ── Metadata store ───────────────────────────────────────────
    let metadata_path = idx_dir.join(METADATA_DB);
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;
    // Clear previous data (fresh index).
    metadata_store
        .clear()
        .context("failed to clear metadata store")?;
    metadata_store
        .insert_chunks(chunks)
        .context("failed to insert chunk metadata")?;

    // Store file content hashes for incremental indexing.
    for (file_path, hash) in file_hashes {
        metadata_store
            .set_file_hash(file_path, hash)
            .context("failed to store file hash")?;
    }

    // Store call-site references for call graph analysis.
    for (file_path, refs) in file_refs {
        metadata_store
            .insert_references(file_path, refs)
            .context("failed to store references")?;
    }

    metadata_store
        .set_index_meta("model_name", model_name)
        .context("failed to store model_name")?;
    metadata_store
        .set_index_meta("embedding_dim", &dim.to_string())
        .context("failed to store embedding_dim")?;

    debug!(chunks = chunks.len(), "metadata stored");

    // ── Vector store ─────────────────────────────────────────────
    let vector_path = idx_dir.join(VECTOR_DB);
    let vector_store =
        VectorStore::open(&vector_path, dim).context("failed to open vector store")?;
    vector_store
        .clear()
        .context("failed to clear vector store")?;

    let batch: Vec<(&str, &[f32])> = embeddings
        .iter()
        .map(|(id, vec)| (id.as_str(), vec.as_slice()))
        .collect();
    vector_store
        .insert_batch(&batch)
        .context("failed to insert vectors")?;

    debug!(vectors = embeddings.len(), "vectors stored");

    // ── BM25 index ───────────────────────────────────────────────
    let bm25_dir = idx_dir.join(BM25_SUBDIR);
    let bm25_index = Bm25Index::open(&bm25_dir).context("failed to open BM25 index")?;
    bm25_index.clear().context("failed to clear BM25 index")?;

    // Pre-compute language strings so BM25 documents can borrow them.
    let lang_strings: Vec<String> = chunks.iter().map(|c| c.language.to_string()).collect();
    let bm25_docs: Vec<Bm25Document<'_>> = chunks
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
        .context("failed to insert BM25 documents")?;

    debug!(docs = bm25_docs.len(), "BM25 index built");

    Ok(())
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
