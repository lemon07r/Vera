# Agent Usage

Vera is a semantic code search CLI. For the full skill definition, see [`skills/vera/SKILL.md`](skills/vera/SKILL.md).

## Quick Start

```bash
bunx @vera-ai/cli install      # install binary (or: npx -y @vera-ai/cli install)
vera index .                    # index the repo (add .vera/ to .gitignore)
vera search "query"              # search. returns ranked markdown codeblocks
vera update .                   # after code changes
```

## When to Use

- **Vera search**: semantic search, symbol lookup, cross-file discovery, ranked context
- **Vera search --deep**: multi-hop search that follows symbols from initial results for broader exploration
- **Vera grep**: regex pattern search scoped to indexed files (respects .gitignore/.veraignore)
- **rg**: exact text, regex, bulk find-and-replace across all files

## Output

Default output is markdown codeblocks with file path, line range, and optional symbol info in the info string. This is the most token-efficient format for reading results as context. Use `--json` for compact single-line JSON (programmatic consumption or piping to other tools), or `--raw` for verbose human-readable output with all fields. Use `--timing` to print pipeline step durations to stderr.

## References

Query tips, troubleshooting, MCP setup, and install details are in [`skills/vera/SKILL.md`](skills/vera/SKILL.md) and its `references/` subdirectory.
