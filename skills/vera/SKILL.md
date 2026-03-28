---
name: vera
description: Semantic code search, regex pattern search, and symbol lookup across a local repository. Returns ranked markdown codeblocks with file path, line range, content, and optional symbol info. Use when the user asks to find where logic lives, what calls a function, how a feature is implemented, which files handle a concept, or wants to explore unfamiliar code by intent. Also use for symbol lookup when the exact name appears in many files. Use `vera grep` for regex pattern matching scoped to indexed files. Use rg for bulk find-and-replace or searching outside the index.
---

# Vera

Semantic code search CLI. Combines BM25 keyword matching with vector similarity and cross-encoder reranking to return the most relevant code for a natural-language query.

## Workflow

1. Ensure Vera is installed and on `PATH` (add `.vera/` to `.gitignore` on first use). If missing: `references/install.md`.
2. Index the repo: `vera index .` (first time) or `vera update .` (after edits).
3. Get oriented: `vera overview` returns a project summary (languages, directories, entry points, hotspots).
4. Trace references: `vera references <symbol>` finds callers; add `--callees` to see what it calls. `vera dead-code` lists functions with no callers.
5. Search:
   ```sh
   vera search "authentication middleware"
   vera search "parse_config" --type function --limit 5
   vera search "database connection" --lang rust --path "src/**"
   vera search "config loading" --deep    # multi-hop: follows symbols from initial results
   ```
6. Regex search (exact patterns, imports, TODOs):
   ```sh
   vera grep "fn\s+main"
   vera grep "TODO|FIXME" -i
   vera grep "use std::collections" --context 0
   ```
7. Use the first results (they are ranked by relevance). Output is markdown codeblocks by default.

## Example Output

```sh
vera search "hybrid search" --limit 1
```

````
```crates/vera-core/src/retrieval/hybrid.rs:58-110 function:search_hybrid
pub async fn search_hybrid(...) -> Result<Vec<SearchResult>> { ... }
```
````

The info string contains `file_path:line_start-line_end` and optional `symbol_type:symbol_name`. Use `--json` for compact single-line JSON (programmatic consumption), or `--raw` for verbose human-readable output. Use `--timing` to print pipeline step durations to stderr.

## Query Strategy

- Describe behavior or intent: "JWT token validation", "request rate limiting", not "code" or "utils".
- For known symbol names, search the exact name: `vera search "parse_config"`.
- Start broad, then narrow with `--lang`, `--path`, `--type`, `--limit`.
- After code changes mid-session, run `vera update .` before searching again.
- Use `vera grep` for regex patterns scoped to indexed files (respects .gitignore/.veraignore).
- Use `vera search --deep` when initial results need broader context (follows symbols from first results).
- Use `rg` for bulk find-and-replace or searching outside the index.

## Failure Recovery

- `no index found` → `vera index .`
- stale results after edits → `vera update .`
- local model/ONNX fails → `vera doctor`, then `references/troubleshooting.md`
- API credentials missing → `references/install.md`
- MCP requested → `references/mcp.md`

## References

- `references/install.md`: install, setup, API and local config
- `references/query-patterns.md`: more query examples and rg guidance
- `references/troubleshooting.md`: common errors and fixes
- `references/mcp.md`: optional MCP server usage
