---
name: vera
description: Semantic code search, regex pattern search, and symbol lookup across a local repository. Returns ranked markdown codeblocks with file path, line range, content, and optional symbol info. Use `vera search` for conceptual/behavioral queries (how a feature works, where logic lives, exploring unfamiliar code). Use `vera grep` for exact strings, regex patterns, imports, and TODOs. Use `vera references` to trace callers/callees. Use rg only for bulk find-and-replace or files outside the index.
---

# Vera

Semantic code search CLI. Combines BM25 keyword matching with vector similarity and cross-encoder reranking to return the most relevant code for a natural-language query.

## Workflow

1. Ensure Vera is installed and on `PATH` (add `.vera/` to `.gitignore` on first use). If missing: `references/install.md`.
2. Configure exclusions: Vera respects `.gitignore` by default (no setup needed). If you need custom exclusions, create a `.veraignore` file (gitignore syntax). Important: `.veraignore` **replaces** `.gitignore` rules entirely. To keep gitignore rules and add your own on top, start `.veraignore` with `#include .gitignore`, then add only your extra patterns (do not repeat entries already in `.gitignore`). Use `--exclude` for one-off exclusions, `--no-ignore` to disable all ignore parsing, or `--verbose` during indexing to see which files are skipped.
3. Index the repo: `vera index .` (first time) or `vera update .` (after edits). Use `vera index . --verbose` to debug exclusion rules.
4. For long sessions, start the watcher: `vera watch .` (background process, Ctrl-C to stop, 2s debounce). This auto-updates the index on file changes and replaces manual `vera update .` calls.
5. Get oriented: `vera overview` returns a project summary: language breakdown, directory structure, entry points, complexity hotspots, and detected conventions (frameworks, patterns, config files). Use this for onboarding before searching.
6. Use `vera references <symbol>` to find callers; add `--callees` to see what it calls. `vera dead-code` lists functions with no callers.
7. Search:
   ```sh
   vera search "authentication middleware"
   vera search "parse_config" --type function --limit 5
   vera search "database connection" --lang rust --path "src/**"
   vera search "keybind handling" --scope docs
   vera search "mod loader" --scope runtime --include-generated
   vera search "config loading" --deep    # RAG-fusion query expansion (falls back to iterative symbol-following)
   vera search "auth" --compact            # signatures only: broad exploration in fewer tokens
   ```
8. Regex search (exact patterns, imports, TODOs). `vera grep` only searches indexed files, so `.veraignore` and exclusion rules apply:
   ```sh
   vera grep "fn\s+main"
   vera grep "TODO|FIXME" -i              # case-insensitive
   vera grep "keybind" --scope docs        # scoped to docs
   vera grep "use std::collections" --context 0  # no surrounding lines
   vera grep "handler" --compact           # signatures only
   ```
9. Use the first results (they are ranked by relevance). Output is markdown codeblocks by default.

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

## Choosing the Right Tool

| Need | Tool |
|------|------|
| Concepts, behavior, "how does X work" | `vera search` |
| Exact strings, regex, imports, TODOs | `vera grep` |
| Bulk find-and-replace, files outside index | `rg` |

`vera search` understands synonyms and related concepts. `vera grep` matches literal patterns.

## Search Scopes

| Scope | What it includes |
|-------|------------------|
| `source` | Application source code (default bias) |
| `docs` | Markdown, READMEs, ADRs, guides |
| `runtime` | Extracted runtime trees, bundled app code |
| `all` | Everything, no filtering |

Vera favors source files by default. Use `--scope docs` for prose and ADRs, `--scope runtime` for extracted bundles, and `--include-generated` for minified/dist artifacts.

## Search Modes

- **Default**: full results with code bodies. Best for targeted retrieval ("how does BM25 scoring work?").
- **`--deep`**: deep search. When a completion endpoint is configured (`VERA_COMPLETION_BASE_URL`), expands the query into multiple variants via an LLM and fuses results with RRF. Otherwise falls back to iterative symbol-following search. Use when initial results need broader context.
- **`--compact`**: signatures only (name, parameters, return type). Fits more results into fewer tokens. Best for broad exploration ("what functions handle auth?"). Works with `vera grep` too.

## Query Strategy

- Describe behavior or intent: "JWT token validation", "request rate limiting", not "code" or "utils".
- Avoid overly broad queries like "authentication" or "tools". Be specific about what aspect you need.
- Match your intent to the query: for documentation, use doc-focused keywords ("setup guide", "configuration README"); for code, use implementation terms ("token refresh logic", "error handling implementation").
- Use 2-3 varied queries to capture different aspects (e.g., "OAuth token refresh", "JWT expiry handling", "auth middleware"). Results are deduplicated and reranked together.
- Add an `intent` parameter to describe your higher-level goal when the query alone is ambiguous (e.g., query: "config", intent: "find where database connection strings are loaded from environment variables").
- For known symbol names, search the exact name: `vera search "parse_config"`.
- Start broad, then narrow with `--lang`, `--path`, `--type`, `--limit`.
- After code changes mid-session, run `vera update .` before searching again (or use `vera watch .` to auto-update).

## Failure Recovery

- `no index found` → `vera index .`
- stale results after edits → `vera update .`
- local model/ONNX fails → `vera doctor --probe`, then `references/troubleshooting.md`
- missing local assets → `vera repair`
- switch GPU/model backend → `vera backend`
- API credentials missing → `references/install.md`
- MCP requested → `references/mcp.md`

## References

- `references/install.md`: install, setup, API and local config
- `references/query-patterns.md`: more query examples and rg guidance
- `references/troubleshooting.md`: common errors and fixes
- `references/mcp.md`: optional MCP server usage
