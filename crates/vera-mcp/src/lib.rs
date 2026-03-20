//! Vera MCP Server — Model Context Protocol interface for Vera.
//!
//! Exposes Vera's indexing and retrieval capabilities as MCP tools over
//! JSON-RPC 2.0 with stdio transport:
//! - `search_code` — search the indexed codebase
//! - `index_project` — trigger indexing of a project
//! - `update_project` — trigger incremental index update
//! - `get_stats` — retrieve index statistics
//!
//! # Usage
//!
//! Start the server with `vera mcp`. The server reads JSON-RPC messages
//! from stdin and writes responses to stdout. Logs go to stderr.

pub mod protocol;
pub mod server;
pub mod tools;
