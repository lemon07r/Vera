//! Tests for incremental update logic.

use std::fs;

use tempfile::TempDir;

use crate::config::VeraConfig;
use crate::embedding::test_helpers::MockProvider;
use crate::indexing::{index_dir, index_repository, update_repository};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;

use super::content_hash;

fn default_config() -> VeraConfig {
    VeraConfig::default()
}

// ── Content hash tests ──────────────────────────────────────────────

#[test]
fn content_hash_deterministic() {
    let h1 = content_hash("fn main() {}");
    let h2 = content_hash("fn main() {}");
    assert_eq!(h1, h2);
}

#[test]
fn content_hash_different_content() {
    let h1 = content_hash("fn main() {}");
    let h2 = content_hash("fn main() { println!(\"hi\"); }");
    assert_ne!(h1, h2);
}

#[test]
fn content_hash_is_hex_sha256() {
    let h = content_hash("hello");
    // SHA-256 hex is 64 characters.
    assert_eq!(h.len(), 64);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

// ── Update: no changes ──────────────────────────────────────────────

#[tokio::test]
async fn update_no_changes() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    let idx_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    assert!(idx_summary.chunks_created > 0);

    // Update with no changes.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);
    assert!(update_summary.files_unchanged > 0);
    assert_eq!(
        update_summary.total_chunks,
        idx_summary.chunks_created as u64
    );
}

// ── Update: file modified ───────────────────────────────────────────

#[tokio::test]
async fn update_modified_file() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    fs::write(dir.path().join("lib.py"), "def greet():\n    print('hi')\n").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    let idx_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    let _initial_chunks = idx_summary.chunks_created as u64;

    // Modify one file.
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"updated content\");\n}\n\nfn helper() {\n    // new function\n}\n",
    )
    .unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);
    assert_eq!(update_summary.files_unchanged, 1); // lib.py unchanged

    // Verify updated content is in the index.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("updated content", 10).unwrap();
    assert!(!results.is_empty(), "BM25 should find updated content");

    // Verify helper function is searchable.
    let results = bm25.search("helper", 10).unwrap();
    assert!(!results.is_empty(), "BM25 should find new function");
}

#[tokio::test]
async fn update_detects_rst_include_dependency_changes() {
    let dir = TempDir::new().unwrap();
    let docs = dir.path().join("docs");
    let includes = docs.join("includes");
    fs::create_dir_all(&includes).unwrap();

    fs::write(
        docs.join("index.rst"),
        "Guide\n=====\n\n.. include:: includes/common.rst.inc\n",
    )
    .unwrap();
    fs::write(
        includes.join("common.rst.inc"),
        "Original include fragment text.\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let indexed_files = metadata.indexed_files().unwrap();
    assert!(indexed_files.contains(&"docs/index.rst".to_string()));
    assert!(
        !indexed_files.contains(&"docs/includes/common.rst.inc".to_string()),
        "rst include fragments should not be indexed as standalone files"
    );

    fs::write(
        includes.join("common.rst.inc"),
        "Updated include fragment text from dependency.\n",
    )
    .unwrap();

    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);

    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25
        .search("Updated include fragment text from dependency", 10)
        .unwrap();
    assert!(
        !results.is_empty(),
        "BM25 should reflect updated included RST content"
    );
}

// ── Update: file added ──────────────────────────────────────────────

#[tokio::test]
async fn update_added_file() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    let idx_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    let initial_chunks = idx_summary.chunks_created as u64;

    // Add a new file.
    fs::write(
        dir.path().join("utils.py"),
        "def utility_function():\n    return 42\n",
    )
    .unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 0);
    assert!(
        update_summary.total_chunks > initial_chunks,
        "total chunks should increase after adding a file"
    );

    // Verify new file content is searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("utility_function", 10).unwrap();
    assert!(
        !results.is_empty(),
        "BM25 should find content from new file"
    );

    // Verify stats: file count should have increased.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(files.contains(&"utils.py".to_string()));
}

#[tokio::test]
async fn update_replaces_type_relations_for_modified_file() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("types.ts"),
        "interface Loader {}\nclass Repo implements Loader {\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    fs::write(
        dir.path().join("types.ts"),
        "interface Saver {}\nclass Repo implements Saver {\n}\n",
    )
    .unwrap();

    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let store = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert!(store.find_type_relations("Loader").unwrap().is_empty());
    let saver = store.find_type_relations("Saver").unwrap();
    assert_eq!(saver.len(), 1);
    assert_eq!(saver[0].owner, "Repo");
}

// ── Update: file deleted ────────────────────────────────────────────

#[tokio::test]
async fn update_deleted_file() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    fs::write(dir.path().join("lib.py"), "def greet():\n    print('hi')\n").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    let idx_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    let initial_chunks = idx_summary.chunks_created as u64;

    // Delete a file.
    fs::remove_file(dir.path().join("lib.py")).unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 1);
    assert!(
        update_summary.total_chunks < initial_chunks,
        "total chunks should decrease after deleting a file"
    );

    // Verify deleted content is no longer searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("greet", 10).unwrap();
    assert!(
        results.is_empty(),
        "BM25 should not find content from deleted file"
    );

    // Verify stats: file count should have decreased.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(!files.contains(&"lib.py".to_string()));
}

// ── Update: no index exists ─────────────────────────────────────────

#[tokio::test]
async fn update_without_index_fails() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    let result = update_repository(dir.path(), &provider, &config, "mock-model").await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no index found"),
        "should report no index: {err}"
    );
}

// ── Update: consistency with fresh index ────────────────────────────

#[tokio::test]
async fn update_matches_fresh_index() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    fs::write(dir.path().join("lib.py"), "def greet():\n    print('hi')\n").unwrap();
    fs::write(
        dir.path().join("config.toml"),
        "[package]\nname = \"test\"\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    // Apply a sequence of changes.
    // 1. Modify main.rs
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"updated\");\n}\nfn helper() { 42 }\n",
    )
    .unwrap();
    // 2. Delete lib.py
    fs::remove_file(dir.path().join("lib.py")).unwrap();
    // 3. Add a new file
    fs::write(
        dir.path().join("utils.rs"),
        "pub fn utility() -> i32 {\n    42\n}\n",
    )
    .unwrap();

    // Run update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 1);

    // Get stats after update.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let updated_metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let updated_chunk_count = updated_metadata.chunk_count().unwrap();
    let updated_files = updated_metadata.indexed_files().unwrap();
    let updated_languages = updated_metadata.language_stats().unwrap();

    // Now do a fresh index of the same directory.
    let _fresh_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let fresh_metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let fresh_chunk_count = fresh_metadata.chunk_count().unwrap();
    let fresh_files = fresh_metadata.indexed_files().unwrap();
    let fresh_languages = fresh_metadata.language_stats().unwrap();

    // Verify consistency.
    assert_eq!(
        updated_chunk_count, fresh_chunk_count,
        "chunk count should match fresh index"
    );
    assert_eq!(
        updated_files, fresh_files,
        "indexed files should match fresh index"
    );
    assert_eq!(
        updated_languages, fresh_languages,
        "language stats should match fresh index"
    );
}

// ── Update: mixed operations ────────────────────────────────────────

#[tokio::test]
async fn update_mixed_add_modify_delete() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
    fs::write(dir.path().join("b.py"), "def b(): pass").unwrap();
    fs::write(dir.path().join("c.go"), "func c() {}").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    // Modify a.rs, delete b.py, add d.ts.
    fs::write(dir.path().join("a.rs"), "fn a_updated() { 42 }").unwrap();
    fs::remove_file(dir.path().join("b.py")).unwrap();
    fs::write(dir.path().join("d.ts"), "function d(): void {}").unwrap();

    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 1);
    assert_eq!(update_summary.files_unchanged, 1); // c.go

    // Verify the modified content is searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("a_updated", 10).unwrap();
    assert!(!results.is_empty(), "should find updated function name");

    // Verify deleted content is gone.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(!files.contains(&"b.py".to_string()));
    assert!(files.contains(&"d.ts".to_string()));
}

// ── Update: vector store consistency ────────────────────────────────

#[tokio::test]
async fn update_vector_store_consistent() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() { println!(\"hello\"); }",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    // Initial index.
    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let initial_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();

    // Add a file.
    fs::write(dir.path().join("lib.py"), "def lib(): return 1").unwrap();
    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let after_add_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();
    assert!(
        after_add_vec_count > initial_vec_count,
        "vector count should increase after adding a file"
    );

    // Delete the added file.
    fs::remove_file(dir.path().join("lib.py")).unwrap();
    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let after_del_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();
    assert_eq!(
        after_del_vec_count, initial_vec_count,
        "vector count should return to initial after delete"
    );

    // Verify metadata and vectors are in sync.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert_eq!(
        metadata.chunk_count().unwrap(),
        after_del_vec_count,
        "metadata chunk count should match vector count"
    );
}
