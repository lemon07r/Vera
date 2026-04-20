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

## Git-Scoped Search

When the task is limited to your working tree or a PR diff, scope the search before broadening the query:

```bash
vera search "token validation" --changed
vera grep "TODO|FIXME" --changed
vera overview --base origin/main
```

Use:

- `--changed` for modified, staged, and untracked files
- `--since <rev>` for changes since a specific revision
- `--base <rev>` for changes since `merge-base(HEAD, <rev>)`

## Multi-Query Search

If one phrasing is too narrow, pass 2-3 varied queries in one call:

```bash
vera search "OAuth token refresh" "JWT expiry handling" "auth middleware"
```

Vera runs each query, then merges the results with reciprocal rank fusion.

## Intent

If the query is short but your goal is specific, add `--intent`:

```bash
vera search "config" --intent "find where database connection strings are loaded from environment variables"
```

This works best when the query alone is too ambiguous to steer ranking.

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

## When to Use `vera grep` vs `rg`

Use `vera grep` when you want exact text or regex matches limited to indexed files:

- `vera grep "EMBEDDING_MODEL_BASE_URL"`
- `vera grep "TODO\(" -i`
- `vera grep "queryClient|invalidateQueries" --path "frontend/src/**"`

Use `rg` when you need:

- Counting occurrences
- Find-and-replace prep
- File name search
- Files outside the Vera index

## Structural Search

Use `vera ast-query` when you know the AST shape you need and regex would be too blunt:

```bash
vera ast-query '(function_item name: (identifier) @fn)' --lang rust
vera ast-query '(function_definition name: (identifier) @fn)' --lang python --path "src/**"
```

This is raw tree-sitter query syntax, not a Semgrep-style rule DSL.

## Missing Files Or Surprising Exclusions

If a file is missing from search results and you need the exact reason, ask Vera directly:

```bash
vera explain-path path/to/file
```

Use `vera stats --json` when you want the repo-wide health view for parse failures, tree-sitter errors, and Tier 0 fallback.

## Output Format

See [features: output formats](features.md#multiple-output-formats) for all options (`--json`, `--raw`, `--timing`). `--raw` and `--timing` work with `vera search` and `vera grep`, and can appear before or after the subcommand. `vera search --timing` prints per-stage timings; `vera grep --timing` prints total regex-search time.

## Keeping Results Fresh

If results feel stale after code changes, run `vera update .`. See [troubleshooting](troubleshooting.md#results-feel-stale-or-outdated) for details.
