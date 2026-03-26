//! SQLite-based metadata store for chunk attributes.
//!
//! Stores chunk metadata (file path, line ranges, language, symbol info)
//! in a SQLite database. Uses WAL mode for concurrent read performance.

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::types::{Chunk, Language, SymbolType};

/// SQLite-backed metadata store for chunk attributes.
pub struct MetadataStore {
    conn: Connection,
}

impl MetadataStore {
    /// Open (or create) a metadata store at the given path.
    pub fn open(db_path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("failed to open metadata db: {}", db_path.display()))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory metadata store (useful for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory metadata db")?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema and pragmas.
    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA foreign_keys=ON;",
            )
            .context("failed to set SQLite pragmas")?;

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT NOT NULL,
                    symbol_type TEXT,
                    symbol_name TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path
                    ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_language
                    ON chunks(language);
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name
                    ON chunks(symbol_name);",
            )
            .context("failed to create chunks table")?;

        // File-level content hashing for incremental indexing.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL
                );",
            )
            .context("failed to create file_hashes table")?;

        // Index metadata (model name, dimensions, etc.)
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );",
            )
            .context("failed to create index_metadata table")?;

        Ok(())
    }

    /// Insert a batch of chunks into the store.
    pub fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin transaction")?;
        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "INSERT OR REPLACE INTO chunks
                     (id, file_path, line_start, line_end, content, language, symbol_type, symbol_name)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )
                .context("failed to prepare insert statement")?;

            for chunk in chunks {
                let sym_type = chunk.symbol_type.map(|st| st.to_string());
                stmt.execute(params![
                    chunk.id,
                    chunk.file_path,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.content,
                    chunk.language.to_string(),
                    sym_type,
                    chunk.symbol_name,
                ])
                .with_context(|| format!("failed to insert chunk: {}", chunk.id))?;
            }
        }
        tx.commit().context("failed to commit chunk inserts")?;
        Ok(())
    }

    /// Get a chunk by its ID.
    pub fn get_chunk(&self, id: &str) -> Result<Option<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks WHERE id = ?1",
            )
            .context("failed to prepare select statement")?;

        let result = stmt
            .query_row(params![id], |row| Ok(row_to_chunk(row)))
            .optional()
            .context("failed to query chunk by id")?;

        match result {
            Some(chunk) => Ok(Some(chunk?)),
            None => Ok(None),
        }
    }

    /// Get all chunks for a given file path.
    pub fn get_chunks_by_file(&self, file_path: &str) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks WHERE file_path = ?1
                 ORDER BY line_start",
            )
            .context("failed to prepare file chunks query")?;

        let rows = stmt
            .query_map(params![file_path], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by file")?;

        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row.context("failed to read chunk row")??);
        }
        Ok(chunks)
    }

    /// Get all chunks whose symbol name matches exactly (case-insensitive).
    pub fn get_chunks_by_symbol_name(&self, symbol_name: &str) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE lower(symbol_name) = lower(?1)
                 ORDER BY file_path, line_start",
            )
            .context("failed to prepare symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by symbol name")?;

        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row.context("failed to read symbol chunk row")??);
        }
        Ok(chunks)
    }

    /// Get all chunks whose symbol name matches exactly (case-sensitive).
    pub fn get_chunks_by_symbol_name_case_sensitive(&self, symbol_name: &str) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE symbol_name = ?1
                 ORDER BY file_path, line_start",
            )
            .context("failed to prepare case-sensitive symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by symbol name (case-sensitive)")?;

        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row.context("failed to read case-sensitive symbol chunk row")??);
        }
        Ok(chunks)
    }

    /// Get chunks whose symbol names contain the given term (case-insensitive).
    pub fn get_chunks_by_symbol_name_substring(
        &self,
        symbol_name: &str,
        limit: usize,
    ) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE symbol_name IS NOT NULL
                   AND instr(lower(symbol_name), lower(?1)) > 0
                 ORDER BY file_path, line_start
                 LIMIT ?2",
            )
            .context("failed to prepare substring symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name, limit as i64], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by symbol name substring")?;

        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row.context("failed to read substring symbol chunk row")??);
        }
        Ok(chunks)
    }

    /// Count total chunks in the store.
    pub fn chunk_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .context("failed to count chunks")?;
        Ok(count as u64)
    }

    /// Delete all chunks associated with a file path.
    pub fn delete_chunks_by_file(&self, file_path: &str) -> Result<u64> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM chunks WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete chunks by file")?;
        Ok(deleted as u64)
    }

    /// Clear all data from the store.
    pub fn clear(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "DELETE FROM chunks; DELETE FROM file_hashes; DELETE FROM index_metadata;",
            )
            .context("failed to clear metadata store")?;
        Ok(())
    }

    /// Store a key-value pair in index_metadata.
    pub fn set_index_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .context("failed to set index metadata")?;
        Ok(())
    }

    /// Retrieve a key's value from index_metadata.
    pub fn get_index_meta(&self, key: &str) -> Result<Option<String>> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM index_metadata WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .context("failed to get index metadata")?;
        Ok(result)
    }

    /// Store a file content hash for incremental indexing.
    pub fn set_file_hash(&self, file_path: &str, hash: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO file_hashes (file_path, content_hash)
                 VALUES (?1, ?2)",
                params![file_path, hash],
            )
            .context("failed to set file hash")?;
        Ok(())
    }

    /// Get the stored content hash for a file.
    pub fn get_file_hash(&self, file_path: &str) -> Result<Option<String>> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?1",
                params![file_path],
                |row| row.get(0),
            )
            .optional()
            .context("failed to get file hash")?;
        Ok(result)
    }

    /// Delete a file hash entry.
    pub fn delete_file_hash(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM file_hashes WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete file hash")?;
        Ok(())
    }

    /// Get distinct file paths in the index.
    pub fn indexed_files(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT file_path FROM chunks ORDER BY file_path")
            .context("failed to prepare indexed files query")?;
        let rows = stmt
            .query_map([], |row| row.get(0))
            .context("failed to query indexed files")?;
        let mut files = Vec::new();
        for row in rows {
            files.push(row.context("failed to read file path")?);
        }
        Ok(files)
    }

    /// Count distinct files in the index.
    pub fn file_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(DISTINCT file_path) FROM chunks", [], |row| {
                row.get(0)
            })
            .context("failed to count files")?;
        Ok(count as u64)
    }

    /// Get language breakdown (language -> chunk count).
    pub fn language_stats(&self) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT language, COUNT(*) FROM chunks
                 GROUP BY language ORDER BY COUNT(*) DESC",
            )
            .context("failed to prepare language stats query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query language stats")?;
        let mut stats = Vec::new();
        for row in rows {
            let (lang, count) = row.context("failed to read language stat")?;
            stats.push((lang, count as u64));
        }
        Ok(stats)
    }
}

/// Convert a SQLite row into a Chunk.
fn row_to_chunk(row: &rusqlite::Row<'_>) -> Result<Chunk> {
    let id: String = row.get(0).context("missing id")?;
    let file_path: String = row.get(1).context("missing file_path")?;
    let line_start: u32 = row.get(2).context("missing line_start")?;
    let line_end: u32 = row.get(3).context("missing line_end")?;
    let content: String = row.get(4).context("missing content")?;
    let language_str: String = row.get(5).context("missing language")?;
    let symbol_type_str: Option<String> = row.get(6).context("missing symbol_type")?;
    let symbol_name: Option<String> = row.get(7).context("missing symbol_name")?;

    let language = parse_language(&language_str);
    let symbol_type = symbol_type_str.as_deref().map(parse_symbol_type);

    Ok(Chunk {
        id,
        file_path,
        line_start,
        line_end,
        content,
        language,
        symbol_type,
        symbol_name,
    })
}

/// Parse a language string back into the enum.
/// Delegates to `Language::from_str()` to stay in sync with the `Display` impl.
fn parse_language(s: &str) -> Language {
    s.parse::<Language>().unwrap_or(Language::Unknown)
}

/// Parse a symbol type string back into the enum.
fn parse_symbol_type(s: &str) -> SymbolType {
    match s {
        "function" => SymbolType::Function,
        "method" => SymbolType::Method,
        "class" => SymbolType::Class,
        "struct" => SymbolType::Struct,
        "enum" => SymbolType::Enum,
        "trait" => SymbolType::Trait,
        "interface" => SymbolType::Interface,
        "type_alias" => SymbolType::TypeAlias,
        "constant" => SymbolType::Constant,
        "variable" => SymbolType::Variable,
        "module" => SymbolType::Module,
        _ => SymbolType::Block,
    }
}

/// Extension trait to make `optional()` work with rusqlite.
trait OptionalExt<T> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error>;
}

impl<T> OptionalExt<T> for std::result::Result<T, rusqlite::Error> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error> {
        match self {
            Ok(val) => Ok(Some(val)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                id: "src/main.rs:0".to_string(),
                file_path: "src/main.rs".to_string(),
                line_start: 1,
                line_end: 5,
                content: "fn main() {\n    println!(\"hello\");\n}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("main".to_string()),
            },
            Chunk {
                id: "src/main.rs:1".to_string(),
                file_path: "src/main.rs".to_string(),
                line_start: 7,
                line_end: 12,
                content: "struct Config {\n    name: String,\n}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Struct),
                symbol_name: Some("Config".to_string()),
            },
            Chunk {
                id: "src/lib.py:0".to_string(),
                file_path: "src/lib.py".to_string(),
                line_start: 1,
                line_end: 3,
                content: "def hello():\n    pass".to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("hello".to_string()),
            },
        ]
    }

    #[test]
    fn insert_and_count() {
        let store = MetadataStore::open_in_memory().unwrap();
        let chunks = sample_chunks();
        store.insert_chunks(&chunks).unwrap();
        assert_eq!(store.chunk_count().unwrap(), 3);
    }

    #[test]
    fn get_chunk_by_id() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunk = store.get_chunk("src/main.rs:0").unwrap().unwrap();
        assert_eq!(chunk.file_path, "src/main.rs");
        assert_eq!(chunk.symbol_name, Some("main".to_string()));
        assert_eq!(chunk.language, Language::Rust);
        assert_eq!(chunk.symbol_type, Some(SymbolType::Function));
        assert_eq!(chunk.line_start, 1);
        assert_eq!(chunk.line_end, 5);
    }

    #[test]
    fn get_nonexistent_chunk() {
        let store = MetadataStore::open_in_memory().unwrap();
        assert!(store.get_chunk("nonexistent").unwrap().is_none());
    }

    #[test]
    fn get_chunks_by_file() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store.get_chunks_by_file("src/main.rs").unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].id, "src/main.rs:0");
        assert_eq!(chunks[1].id, "src/main.rs:1");
    }

    #[test]
    fn get_chunks_by_symbol_name() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store.get_chunks_by_symbol_name("config").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");
    }

    #[test]
    fn get_chunks_by_symbol_name_case_sensitive() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store.get_chunks_by_symbol_name_case_sensitive("Config").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");

        let lower = store.get_chunks_by_symbol_name_case_sensitive("config").unwrap();
        assert!(lower.is_empty());
    }

    #[test]
    fn get_chunks_by_symbol_name_substring() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store
            .get_chunks_by_symbol_name_substring("fig", 10)
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");
    }

    #[test]
    fn delete_chunks_by_file() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();
        assert_eq!(store.chunk_count().unwrap(), 3);

        let deleted = store.delete_chunks_by_file("src/main.rs").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.chunk_count().unwrap(), 1);
    }

    #[test]
    fn clear_store() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();
        store.set_file_hash("src/main.rs", "abc123").unwrap();

        store.clear().unwrap();
        assert_eq!(store.chunk_count().unwrap(), 0);
        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());
    }

    #[test]
    fn file_hash_operations() {
        let store = MetadataStore::open_in_memory().unwrap();

        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());

        store.set_file_hash("src/main.rs", "hash1").unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().unwrap(),
            "hash1"
        );

        // Update hash
        store.set_file_hash("src/main.rs", "hash2").unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().unwrap(),
            "hash2"
        );

        store.delete_file_hash("src/main.rs").unwrap();
        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());
    }

    #[test]
    fn indexed_files() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let files = store.indexed_files().unwrap();
        assert_eq!(files, vec!["src/lib.py", "src/main.rs"]);
    }

    #[test]
    fn language_stats() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let stats = store.language_stats().unwrap();
        assert_eq!(stats.len(), 2);
        // Rust has 2, Python has 1
        assert_eq!(stats[0], ("rust".to_string(), 2));
        assert_eq!(stats[1], ("python".to_string(), 1));
    }

    #[test]
    fn insert_replaces_existing() {
        let store = MetadataStore::open_in_memory().unwrap();
        let mut chunks = sample_chunks();
        store.insert_chunks(&chunks).unwrap();

        // Modify and re-insert
        chunks[0].content = "fn main() { /* updated */ }".to_string();
        store.insert_chunks(&chunks[..1]).unwrap();

        let chunk = store.get_chunk("src/main.rs:0").unwrap().unwrap();
        assert!(chunk.content.contains("updated"));
        assert_eq!(store.chunk_count().unwrap(), 3);
    }

    #[test]
    fn parse_language_roundtrip() {
        // Exhaustive list of ALL Language variants to catch future additions.
        let all_langs = vec![
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Python,
            Language::Go,
            Language::Java,
            Language::C,
            Language::Cpp,
            Language::Ruby,
            Language::Swift,
            Language::Kotlin,
            Language::Scala,
            Language::Zig,
            Language::Lua,
            Language::Bash,
            Language::CSharp,
            Language::Php,
            Language::Haskell,
            Language::Elixir,
            Language::Dart,
            Language::Sql,
            Language::Hcl,
            Language::Protobuf,
            // Tier 1B
            Language::Html,
            Language::Css,
            Language::Scss,
            Language::Vue,
            Language::GraphQl,
            Language::CMake,
            Language::Dockerfile,
            Language::Xml,
            // Tier 2A
            Language::ObjectiveC,
            Language::Perl,
            Language::Julia,
            Language::Nix,
            Language::OCaml,
            Language::Groovy,
            Language::Clojure,
            Language::CommonLisp,
            Language::Erlang,
            Language::FSharp,
            Language::Fortran,
            Language::PowerShell,
            Language::R,
            // Tier 2A batch 2
            Language::Matlab,
            Language::DLang,
            Language::Fish,
            Language::Zsh,
            Language::Luau,
            Language::Scheme,
            Language::Racket,
            Language::Elm,
            Language::Glsl,
            Language::Hlsl,
            // Tier 2B
            Language::Svelte,
            Language::Astro,
            Language::Makefile,
            Language::Ini,
            Language::Nginx,
            Language::Prisma,
            // Tier 0
            Language::Toml,
            Language::Yaml,
            Language::Json,
            Language::Markdown,
            Language::Unknown,
        ];
        for lang in &all_langs {
            let s = lang.to_string();
            assert_eq!(parse_language(&s), *lang, "Failed roundtrip for {s}");
        }
    }

    #[test]
    fn parse_language_unknown_input() {
        assert_eq!(parse_language("nonexistent"), Language::Unknown);
        assert_eq!(parse_language(""), Language::Unknown);
    }
}
