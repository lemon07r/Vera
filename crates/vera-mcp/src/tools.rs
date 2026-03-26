//! MCP tool definitions and handler dispatch.
//!
//! Defines the tools that the Vera MCP server exposes:
//! - `search_code` — search indexed codebase
//! - `index_project` — trigger full indexing
//! - `update_project` — trigger incremental update
//! - `get_stats` — retrieve index statistics

use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::protocol::{ToolCallResult, ToolDefinition};

/// Compact result representation for MCP tool responses.
/// Drops `score` and `language` (inferrable from extension), omits null fields.
#[derive(Serialize)]
struct CompactResult<'a> {
    file_path: &'a str,
    line_start: u32,
    line_end: u32,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_type: Option<&'a vera_core::types::SymbolType>,
}

/// Serialize search results as compact single-line JSON.
fn compact_results_json(
    results: &[vera_core::types::SearchResult],
) -> Result<String, serde_json::Error> {
    let compact: Vec<CompactResult> = results
        .iter()
        .map(|r| CompactResult {
            file_path: &r.file_path,
            line_start: r.line_start,
            line_end: r.line_end,
            content: &r.content,
            symbol_name: r.symbol_name.as_deref(),
            symbol_type: r.symbol_type.as_ref(),
        })
        .collect();
    serde_json::to_string(&compact)
}

/// Return the list of tools the server advertises.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_code".to_string(),
            description: "Search the indexed codebase using hybrid BM25+vector \
                          retrieval. Returns ranked code snippets with file paths, \
                          line numbers, and relevance scores."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keyword or natural language)"
                    },
                    "lang": {
                        "type": "string",
                        "description": "Filter by programming language (e.g., rust, python)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Filter by file path glob (e.g., src/**/*.rs)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Filter by symbol type (function, struct, class, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "index_project".to_string(),
            description: "Index a project directory for code search. Creates a .vera/ \
                          index with BM25 and vector search capabilities."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the project directory to index"
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "update_project".to_string(),
            description: "Incrementally update the index for a project. Only re-indexes \
                          files that have changed since the last index/update."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the project directory to update"
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "get_stats".to_string(),
            description: "Get index statistics: file count, chunk count, index size, \
                          and language breakdown."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the project directory (default: current dir)"
                    }
                }
            }),
        },
    ]
}

/// Dispatch a tool call to the appropriate handler.
///
/// Returns a `ToolCallResult` — either success with JSON content or an
/// error with a descriptive message. This function never panics.
pub fn handle_tool_call(name: &str, arguments: &Value) -> ToolCallResult {
    match name {
        "search_code" => handle_search_code(arguments),
        "index_project" => handle_index_project(arguments),
        "update_project" => handle_update_project(arguments),
        "get_stats" => handle_get_stats(arguments),
        _ => ToolCallResult::error(format!("Unknown tool: {name}")),
    }
}

/// Handle the `search_code` tool.
fn handle_search_code(args: &Value) -> ToolCallResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolCallResult::error("Missing required parameter: query"),
    };

    let filters = vera_core::types::SearchFilters {
        language: args.get("lang").and_then(|v| v.as_str()).map(String::from),
        path_glob: args.get("path").and_then(|v| v.as_str()).map(String::from),
        symbol_type: args
            .get("symbol_type")
            .and_then(|v| v.as_str())
            .map(String::from),
    };

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let backend = vera_core::config::resolve_backend(None);
    let mut config = vera_core::config::VeraConfig::default();
    config.adjust_for_backend(backend);
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);

    // Look for index in current working directory.
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => return ToolCallResult::error(format!("Failed to get working directory: {e}")),
    };
    let index_dir = vera_core::indexing::index_dir(&cwd);

    if !index_dir.exists() {
        return ToolCallResult::error(
            "No index found in current directory. Run index_project first.",
        );
    }

    // Use the shared search service from vera-core.
    let results = match vera_core::retrieval::search_service::execute_search(
        &index_dir,
        query,
        &config,
        &filters,
        result_limit,
        backend,
    ) {
        Ok(r) => r,
        Err(e) => return ToolCallResult::error(format!("Search failed: {e}")),
    };

    match compact_results_json(&results) {
        Ok(json) => ToolCallResult::success(json),
        Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
    }
}

/// Handle the `index_project` tool.
fn handle_index_project(args: &Value) -> ToolCallResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolCallResult::error("Missing required parameter: path"),
    };

    let repo_path = Path::new(path);
    if !repo_path.exists() {
        return ToolCallResult::error(format!("Path does not exist: {path}"));
    }
    if !repo_path.is_dir() {
        return ToolCallResult::error(format!("Path is not a directory: {path}"));
    }

    let backend = vera_core::config::resolve_backend(None);
    let mut config = vera_core::config::VeraConfig::default();
    config.adjust_for_backend(backend);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return ToolCallResult::error(format!("Failed to create runtime: {e}")),
    };

    let (provider, model_name) = match rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    )) {
        Ok(res) => res,
        Err(e) => {
            return ToolCallResult::error(format!("Failed to create embedding provider: {e}"));
        }
    };

    match rt.block_on(vera_core::indexing::index_repository(
        repo_path,
        &provider,
        &config,
        &model_name,
    )) {
        Ok(summary) => match serde_json::to_string_pretty(&summary) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize summary: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Indexing failed: {e}")),
    }
}

/// Handle the `update_project` tool.
fn handle_update_project(args: &Value) -> ToolCallResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolCallResult::error("Missing required parameter: path"),
    };

    let repo_path = Path::new(path);
    if !repo_path.exists() {
        return ToolCallResult::error(format!("Path does not exist: {path}"));
    }
    if !repo_path.is_dir() {
        return ToolCallResult::error(format!("Path is not a directory: {path}"));
    }

    let backend = vera_core::config::resolve_backend(None);
    let mut config = vera_core::config::VeraConfig::default();
    config.adjust_for_backend(backend);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return ToolCallResult::error(format!("Failed to create runtime: {e}")),
    };

    let (provider, model_name) = match rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    )) {
        Ok(res) => res,
        Err(e) => {
            return ToolCallResult::error(format!("Failed to create embedding provider: {e}"));
        }
    };

    match rt.block_on(vera_core::indexing::update_repository(
        repo_path,
        &provider,
        &config,
        &model_name,
    )) {
        Ok(summary) => match serde_json::to_string_pretty(&summary) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize summary: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Update failed: {e}")),
    }
}

/// Handle the `get_stats` tool.
fn handle_get_stats(args: &Value) -> ToolCallResult {
    let path = args.get("path").and_then(|v| v.as_str());

    let repo_path = match path {
        Some(p) => std::path::PathBuf::from(p),
        None => match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => {
                return ToolCallResult::error(format!("Failed to get working directory: {e}"));
            }
        },
    };

    if !repo_path.exists() {
        return ToolCallResult::error(format!("Path does not exist: {}", repo_path.display()));
    }

    match vera_core::stats::collect_stats(&repo_path) {
        Ok(stats) => match serde_json::to_string_pretty(&stats) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize stats: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to collect stats: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_has_four_tools() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"index_project"));
        assert!(names.contains(&"update_project"));
        assert!(names.contains(&"get_stats"));
    }

    #[test]
    fn tool_definitions_have_valid_schemas() {
        let tools = tool_definitions();
        for tool in &tools {
            let schema = &tool.input_schema;
            assert_eq!(schema["type"], "object", "tool {} schema type", tool.name);
        }
    }

    #[test]
    fn handle_unknown_tool_returns_error() {
        let result = handle_tool_call("nonexistent", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(result.content[0].text.contains("Unknown tool"));
    }

    #[test]
    fn search_code_missing_query_returns_error() {
        let result = handle_tool_call("search_code", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(
            result.content[0]
                .text
                .contains("Missing required parameter")
        );
    }

    #[test]
    fn index_project_missing_path_returns_error() {
        let result = handle_tool_call("index_project", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(
            result.content[0]
                .text
                .contains("Missing required parameter")
        );
    }

    #[test]
    fn update_project_missing_path_returns_error() {
        let result = handle_tool_call("update_project", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(
            result.content[0]
                .text
                .contains("Missing required parameter")
        );
    }

    #[test]
    fn index_project_invalid_path_returns_error() {
        let result = handle_tool_call(
            "index_project",
            &serde_json::json!({"path": "/nonexistent/path/abc"}),
        );
        assert!(result.is_error);
        assert!(result.content[0].text.contains("does not exist"));
    }

    #[test]
    fn update_project_invalid_path_returns_error() {
        let result = handle_tool_call(
            "update_project",
            &serde_json::json!({"path": "/nonexistent/path/abc"}),
        );
        assert!(result.is_error);
        assert!(result.content[0].text.contains("does not exist"));
    }

    #[test]
    fn get_stats_no_index_returns_error() {
        let result = handle_tool_call("get_stats", &serde_json::json!({"path": "/tmp"}));
        assert!(result.is_error);
        // Should mention no index found or similar.
        assert!(!result.content[0].text.is_empty());
    }
}
