# Optional MCP Usage

Vera supports MCP, but MCP is not the preferred install path for agent users. The primary path is:

- install Vera
- install the Vera skill
- use the Vera CLI directly

Use MCP only when:

- the client explicitly requires MCP
- the user asks to integrate Vera through MCP
- the environment already has an MCP-centric workflow

## Start The Server

Wrapper-based commands:

```sh
npx -y @vera-ai/cli mcp
bunx @vera-ai/cli mcp
uvx vera-ai mcp
```

If Vera is already installed on `PATH`:

```sh
vera mcp
```

The server exposes:

- `search_code` (supports `queries` array for multi-query search, `intent` for reranking; auto-indexes and starts watcher on first use)
- `get_stats`
- `get_overview` (includes detected project conventions)
- `regex_search` (kept intentionally minimal: regex, context, scope, generated-file toggle, compact output)

## Guidance

- Keep the Vera skill CLI-centered.
- Keep the MCP surface small. Reach for the CLI skill when you need the full set of filters and commands.
- Only mention MCP when the task or client explicitly depends on it.
- Do not assume MCP is installed if the user only asked for Vera search capability.
