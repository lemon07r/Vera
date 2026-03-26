---
name: vera
description: Semantic code search and symbol lookup across a local repository. Returns ranked code capsules (file path, line range, content, score, symbol info) as JSON. Use when the user asks to find where logic lives, what calls a function, how a feature is implemented, which files handle a concept, or wants to explore unfamiliar code by intent. Also use for symbol lookup when the exact name appears in many files. Do NOT use for exact literal search, regex, counting occurrences, or bulk find-and-replace — use rg for those.
---

# Vera

Semantic code search CLI. Combines BM25 keyword matching with vector similarity and cross-encoder reranking to return the most relevant code for a natural-language query.

## Workflow

1. Ensure Vera is installed and on `PATH` (add `.vera/` to `.gitignore` on first use). If missing: `references/install.md`.
2. Index the repo: `vera index .` (first time) or `vera update .` (after edits).
3. Search:
   ```sh
   vera search "authentication middleware" --json
   vera search "parse_config" --type function --limit 5
   vera search "database connection" --lang rust --path "src/**"
   ```
4. Use the highest-scored results first. Pass `--json` for machine-readable output.

## Example Output

```sh
vera search "hybrid search" --limit 1 --json
```

```json
[{
  "file_path": "crates/vera-core/src/retrieval/hybrid.rs",
  "line_start": 58,
  "line_end": 110,
  "content": "pub async fn search_hybrid(...) -> Result<Vec<SearchResult>> { ... }",
  "language": "rust",
  "score": 2.28,
  "symbol_name": "search_hybrid",
  "symbol_type": "function"
}]
```

Fields: `file_path`, `line_start`, `line_end`, `content`, `language`, `score`, and optional `symbol_name`/`symbol_type`.

## Query Strategy

- Describe behavior or intent: "JWT token validation", "request rate limiting", not "code" or "utils".
- For known symbol names, search the exact name: `vera search "parse_config"`.
- Start broad, then narrow with `--lang`, `--path`, `--type`, `--limit`.
- After code changes mid-session, run `vera update .` before searching again.
- If the user wants exact text or regex, use `rg` instead.

## Failure Recovery

- `no index found` → `vera index .`
- stale results after edits → `vera update .`
- local model/ONNX fails → `vera doctor`, then `references/troubleshooting.md`
- API credentials missing → `references/install.md`
- MCP requested → `references/mcp.md`

## References

- `references/install.md` — install, setup, API and local config
- `references/query-patterns.md` — more query examples and rg guidance
- `references/troubleshooting.md` — common errors and fixes
- `references/mcp.md` — optional MCP server usage
