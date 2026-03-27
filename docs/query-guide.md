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
- `--limit 5`: return fewer results

These stack, so you can combine them:

```bash
vera search "error handling" --lang rust --type function --limit 5
```

## Exact Symbol Names

If you already know the symbol name, search for it directly:

```bash
vera search "parse_config"
vera search "AuthMiddleware"
```

Vera's hybrid pipeline handles both natural language intent and exact symbol lookups.

## When to Use `rg` Instead

Vera is a semantic search tool. For these tasks, use `rg` (ripgrep) or plain grep:

- Exact string matching (`rg "EMBEDDING_MODEL_BASE_URL"`)
- Regex search (`rg "TODO\(" -n`)
- Counting occurrences
- Find-and-replace prep

## Output Format

Output uses markdown codeblocks by default, the most token-efficient format for AI agents. Each result is a fenced codeblock with file path, line range, and optional symbol info in the info string. Use `--json` for compact single-line JSON (programmatic consumption or piping to other tools), or `--raw` for verbose human-readable output with all fields. Use `--timing` to print pipeline step durations to stderr. See the [README](../README.md#usage) for a sample.

## Keeping Results Fresh

If you've changed code since the last index, results may be stale. Run:

```bash
vera update .
```
