# Query Patterns

## Good Vera Queries

```sh
vera search "authentication middleware"
vera search "JWT token validation"
vera search "parse_config"
vera search "request rate limiting" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
```

## Weak Vera Queries

Single generic words return noise:

- `vera search "code"`
- `vera search "utils"`
- `vera search "file"`

Fix: describe what the code *does*, not what it *is*.

## When To Use `vera references` Instead

For structural queries about call relationships, use `references` or `dead-code` instead of `search`:

```sh
vera references parse_config            # who calls parse_config?
vera references parse_config --callees  # what does parse_config call?
vera dead-code                          # functions with no callers
```

These query the call graph built during indexing (direct calls only, no dynamic dispatch).

## When To Use `vera grep` Instead of `rg`

`vera grep` searches only indexed files, so `.veraignore` and exclusion rules apply automatically. Use it when you want results scoped to the project's indexed corpus.

```sh
vera grep "fn\s+main"                      # regex over indexed files
vera grep "TODO|FIXME" -i                   # case-insensitive
vera grep "queryClient|invalidateQueries" --path "frontend/src/**"
vera grep "Authorization" --lang rust --type function
vera grep "handler" --scope docs             # scoped to documentation
vera grep "use std::collections" --context 0 # no surrounding context lines
vera grep "parse" --compact                  # signatures only
```

## When To Use `rg` Instead

- File name search: `rg --files | rg "docker"`
- Counting occurrences
- Bulk find-and-replace prep
- Files outside the Vera index

## Narrowing Results

Add one filter at a time:

1. `--lang rust`: restrict to a language
2. `--path "src/auth/**"`: restrict to a path glob
3. `--type function`: restrict to symbol type
4. `--limit 3`: fewer, higher-confidence results
5. `--scope source`: restrict to a corpus scope (see SKILL.md for scope table)

## Multi-Query Search

`vera search` accepts multiple quoted queries and merges the results with reciprocal rank fusion:

```sh
vera search "OAuth token refresh" "JWT expiry handling" "auth middleware"
```

Use this when one phrasing is too narrow but the task is still one coherent search.

## Intent-Based Reranking

Add `--intent` when the raw query is short but you know the higher-level goal:

```sh
vera search "config" --intent "find where database connection strings are loaded from environment variables"
```

Use this when the raw query is too short or ambiguous to capture what you actually need.
