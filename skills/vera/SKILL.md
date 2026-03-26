---
name: vera
description: Use Vera for semantic code search, symbol lookup, cross-file discovery, and ranked code context in a local repository. Prefer it when the user asks where logic lives, wants related code across files, needs the best matching function/class/module first, or is exploring unfamiliar code by intent rather than exact text. Do not use Vera for exact literal search, regex search, or bulk find-and-replace; use rg for those.
---

# Vera

Use Vera as the agent-first code search tool when the task is about understanding a codebase, not just matching raw text.

Make sure the `vera` CLI is available on `PATH`. The preferred install flow is `npx -y @vera-ai/cli install`, then `vera setup`.

## When To Use Vera

- Semantic search: "authentication logic", "request validation", "database connection pooling"
- Symbol lookup when the exact name might appear in multiple places
- Cross-file discovery: find related code paths, not every raw match
- Ranked context: get the best candidates first instead of a long grep dump

## When Not To Use Vera

- Exact literal or regex matching
- Bulk text replacement
- Counting token occurrences

For those, use `rg`.

## Gitignore

Vera stores its index in a `.vera/` directory at the repository root. **Always ensure `.vera/` is listed in the project's `.gitignore`** before indexing. These files are local, machine-specific, and should never be committed.

## Workflow

1. Make sure Vera is installed. If it is missing, follow `references/install.md`.
2. Make sure `.vera/` is in the project's `.gitignore`. If not, add it.
3. Make sure the repository is indexed.
   - First time: `vera index .`
   - After edits: `vera update .`
4. Run a focused search.
   - Semantic: `vera search "authentication middleware"`
   - Symbol oriented: `vera search "parse_config"`
   - Narrow results: add `--lang`, `--path`, `--type`, `--limit`
5. Use `--json` when you need to parse results or feed them into another tool.

## Query Tips

- Prefer intent phrases over vague nouns.
- Start broad, then add filters when the first pass is noisy.
- If the user gave an exact symbol, search for that exact symbol name first.
- If the user asks for exact text, switch to `rg` instead of forcing Vera.

See `references/query-patterns.md` for more examples.

## Output

Vera returns ranked code capsules with:

- `file_path`
- `line_start` and `line_end`
- `content`
- `language`
- `score`
- optional `symbol_name` and `symbol_type`

Use the highest-ranked results first unless the query clearly needs broad coverage.

## Failure Recovery

- `no index found`: run `vera index <repo-path>`
- results look stale after code changes: run `vera update <repo-path>`
- built-in local model setup fails: see `references/troubleshooting.md`
- API mode is missing credentials: see `references/install.md`
- user asks for MCP specifically: see `references/mcp.md`

## References

- Install and first run: `references/install.md`
- Query examples: `references/query-patterns.md`
- Troubleshooting: `references/troubleshooting.md`
- Optional MCP usage: `references/mcp.md`
