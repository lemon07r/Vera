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

- `search_code`
- `index_project`
- `update_project`
- `get_stats`
- `get_overview`
- `watch_project`
- `find_references`
- `find_dead_code`
- `regex_search`

## Guidance

- Keep the Vera skill CLI-centered.
- Only mention MCP when the task or client explicitly depends on it.
- Do not assume MCP is installed if the user only asked for Vera search capability.
