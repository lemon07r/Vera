# Query Guide

## Writing Good Queries

Vera works best with queries that describe what the code *does*, not just a keyword. Think of it like asking "find me the code that handles X" rather than searching for a variable name.

**Strong queries:**

```bash
vera search "authentication middleware"
vera search "JWT token validation"
vera search "request rate limiting" --lang rust
vera search "database connection pooling"
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
```

**Weak queries:**

```bash
vera search "code"
vera search "utils"
vera search "file"
```

Single generic words return too many results. Be specific about the behavior or concept you're looking for.

## Narrowing Results

If your first search returns too much, add filters one at a time:

- `--lang rust`: restrict to a specific language
- `--path "src/**/*.ts"`: restrict to a file path pattern
- `--type function`: restrict to functions, classes, methods, or structs
- `--scope docs`: restrict to docs and markdown
- `--scope runtime`: restrict to extracted runtime trees and bundled app code
- `--include-generated`: include dist/minified/generated artifacts
- `--limit 5`: return fewer results

These stack, so you can combine them:

```bash
vera search "error handling" --lang rust --type function --limit 5
vera search "mod loader" --scope runtime --include-generated
```

## Exact Symbol Names

If you already know the symbol name, search for it directly:

```bash
vera search "parse_config"
vera search "AuthMiddleware"
```

Vera's hybrid pipeline handles both natural language intent and exact symbol lookups.

## Default Search Bias

Vera favors source files by default. Docs, archives, runtime extracts, and generated files are still available, but they are no longer treated as equally good defaults for everyday coding tasks.

Use `--scope docs` when you are reading guides or ADRs. Use `--scope runtime` when you're debugging extracted app bundles or decompiled runtime code. Add `--include-generated` when you intentionally want minified or generated files in the result set.

## When to Use `rg` Instead

Vera is a semantic search tool. For these tasks, use `rg` (ripgrep) or plain grep:

- Exact string matching (`rg "EMBEDDING_MODEL_BASE_URL"`)
- Regex search (`rg "TODO\(" -n`)
- Counting occurrences
- Find-and-replace prep

## Output Format

See the [README usage section](../README.md#usage) for output format options (`--json`, `--raw`, `--timing`) and a sample.

## Keeping Results Fresh

If results feel stale after code changes, run `vera update .`. See [troubleshooting](troubleshooting.md#results-feel-stale-or-outdated) for details.
