# Vera — Code Search for AI Agents

## Purpose

Vera is a code indexing and retrieval tool built for AI coding agents. It combines BM25 keyword search with vector similarity search (Reciprocal Rank Fusion) and cross-encoder reranking to find relevant code across a codebase. It parses 21+ languages (Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, Ruby, Bash, Kotlin, Swift, Zig, Lua, Scala, C#, PHP, Haskell, Elixir, Dart, SQL, HCL/Terraform, Protobuf) with tree-sitter, chunks at symbol boundaries, and returns structured JSON context capsules with file paths, line ranges, and code content.

## When to Use

**Use Vera instead of grep/ripgrep when you need:**
- Semantic search — find code by intent ("authentication logic") not just keywords
- Symbol discovery — locate function/class/struct definitions by name
- Cross-file understanding — find related code spanning multiple files
- Ranked results — get the most relevant matches first, not every match

**Use grep/ripgrep when you need:**
- Exact string matching (literal text, regex patterns)
- Simple find-and-replace across files
- Counting occurrences of a specific token

## Commands

### Index a project (required first step)
```sh
vera index .                    # Index current directory
vera index /path/to/repo        # Index a specific path
vera index --local .            # Index using local models (no API key needed)
vera index . --json             # JSON output summary
```
Creates a `.vera/` directory with the search index. Requires `EMBEDDING_MODEL_BASE_URL`, `EMBEDDING_MODEL_ID`, and `EMBEDDING_MODEL_API_KEY` environment variables for API mode, or you can use the `--local` flag (or `VERA_LOCAL=1` env var) for local inference (no API key needed).

### Search
```sh
vera search "authentication logic"           # Semantic search
vera search --local "parse_config"           # Search using local models
vera search "error handling" --lang rust     # Filter by language
vera search "routes" --path "src/**/*.ts"    # Filter by path glob
vera search "handler" --type function        # Filter by symbol type
vera search "config" --limit 5 --json        # Limit results, JSON output
```
Searches run from the directory containing `.vera/`. Combine filters freely.

### Update index (after code changes)
```sh
vera update .                   # Re-index only changed files
vera update --local .           # Update using local models
vera update . --json            # JSON output summary
```
Detects added, modified, and deleted files via content hashing. Much faster than full re-index.

### View statistics
```sh
vera stats                      # Human-readable stats
vera stats --json               # JSON output
```
Shows file count, chunk count, index size, and language breakdown.

### MCP server (for tool-use integration)
```sh
vera mcp                        # Start MCP server on stdio
```
Exposes tools: `search_code`, `index_project`, `update_project`, `get_stats`. Same parameters and results as the CLI equivalents.

## Output Interpretation

With `--json`, search returns an array of context capsules:
```json
[
  {
    "file_path": "src/auth/login.rs",
    "line_start": 42,
    "line_end": 68,
    "content": "pub fn authenticate(...) { ... }",
    "language": "rust",
    "score": 0.847,
    "symbol_name": "authenticate",
    "symbol_type": "function"
  }
]
```

**Fields:** `file_path` (repo-relative), `line_start`/`line_end` (1-based, inclusive), `content` (complete symbol body), `language`, `score` (higher = more relevant), `symbol_name` and `symbol_type` (null for non-symbol chunks).

Results are ranked by relevance score, descending. Use `file_path` and line numbers to navigate directly to the code.

## Query Tips

- **Symbol lookup:** Use the exact name — `"parse_config"`, `"UserRepository"`, `"handle_request"`
- **Intent search:** Describe what the code does — `"database connection pooling"`, `"JWT token validation"`
- **Narrow with filters:** Add `--lang`, `--path`, or `--type` to reduce noise in large codebases
- **Start broad, then narrow:** Search without filters first, then add constraints based on what you see
- **Prefer `--json` for programmatic use:** Parse results with `jq` or your language's JSON library
- **Keep the index fresh:** Run `vera update .` after making code changes before searching again
