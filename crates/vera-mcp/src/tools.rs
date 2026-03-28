//! MCP tool definitions and handler dispatch.
//!
//! Defines the tools that the Vera MCP server exposes:
//! - `search_code` — search indexed codebase
//! - `index_project` — trigger full indexing
//! - `update_project` — trigger incremental update
//! - `get_stats` — retrieve index statistics
//! - `get_overview` — architecture overview for agent onboarding
//! - `watch_project` — watch files and auto-update the index
//! - `find_references` — find callers or callees of a symbol
//! - `find_dead_code` — find functions with no callers
//! - `regex_search` — regex search over indexed files

use std::path::Path;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;

use crate::protocol::{ToolCallResult, ToolDefinition};
use crate::watcher::WatchHandle;

/// Global watcher handle. Kept alive for the lifetime of the MCP server process.
static WATCHER: Mutex<Option<WatchHandle>> = Mutex::new(None);

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
                    "scope": {
                        "type": "string",
                        "enum": ["source", "docs", "runtime", "all"],
                        "description": "Coarse corpus scope. Defaults to source-first behavior."
                    },
                    "include_generated": {
                        "type": "boolean",
                        "description": "Include generated or minified files such as dist bundles."
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
        ToolDefinition {
            name: "get_overview".to_string(),
            description: "Get architecture overview of the indexed project: languages, \
                          directories, entry points, symbol types, and complexity hotspots. \
                          Useful for onboarding and understanding project structure."
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
        ToolDefinition {
            name: "watch_project".to_string(),
            description: "Start watching a project directory for file changes and \
                          automatically update the index when files are modified. \
                          Requires an existing index (run index_project first)."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the project directory to watch (default: current dir)"
                    }
                }
            }),
        },
        ToolDefinition {
            name: "find_references".to_string(),
            description: "Find callers or callees of a symbol using the call graph \
                          built during indexing. Returns file paths and line numbers."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to look up (function or method name)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["callers", "callees"],
                        "description": "Whether to find callers (default) or callees"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the project directory (default: current dir)"
                    }
                },
                "required": ["symbol"]
            }),
        },
        ToolDefinition {
            name: "find_dead_code".to_string(),
            description: "Find functions and methods with no callers (potential dead code). \
                          Excludes common entry points like main, new, default, etc."
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
        ToolDefinition {
            name: "regex_search".to_string(),
            description: "Search indexed files using a regex pattern. Returns matches \
                          with surrounding context lines. Useful for exact pattern matching \
                          (imports, TODOs, specific syntax) where semantic search is too broad."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches (default: 20)"
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive matching (default: false)"
                    },
                    "context": {
                        "type": "integer",
                        "description": "Context lines before and after each match (default: 2)"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["source", "docs", "runtime", "all"],
                        "description": "Coarse corpus scope. Defaults to source-first behavior."
                    },
                    "include_generated": {
                        "type": "boolean",
                        "description": "Include generated or minified files such as dist bundles."
                    }
                },
                "required": ["pattern"]
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
        "get_overview" => handle_get_overview(arguments),
        "watch_project" => handle_watch_project(arguments),
        "find_references" => handle_find_references(arguments),
        "find_dead_code" => handle_find_dead_code(arguments),
        "regex_search" => handle_regex_search(arguments),
        _ => ToolCallResult::error(format!("Unknown tool: {name}")),
    }
}

/// Handle the `search_code` tool.
fn handle_search_code(args: &Value) -> ToolCallResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolCallResult::error("Missing required parameter: query"),
    };
    let scope = match args.get("scope").and_then(|v| v.as_str()) {
        Some(value) => match value.parse() {
            Ok(scope) => Some(scope),
            Err(()) => {
                return ToolCallResult::error(format!("Invalid scope: {value}"));
            }
        },
        None => None,
    };

    let filters = vera_core::types::SearchFilters {
        language: args.get("lang").and_then(|v| v.as_str()).map(String::from),
        path_glob: args.get("path").and_then(|v| v.as_str()).map(String::from),
        symbol_type: args
            .get("symbol_type")
            .and_then(|v| v.as_str())
            .map(String::from),
        scope,
        include_generated: Some(
            args.get("include_generated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        ),
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
    let (results, _timings) = match vera_core::retrieval::search_service::execute_search(
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

/// Validate a required path argument and return it as a Path reference.
fn require_dir_path(args: &Value) -> Result<&Path, ToolCallResult> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolCallResult::error("Missing required parameter: path"))?;
    let p = Path::new(path);
    if !p.exists() {
        return Err(ToolCallResult::error(format!(
            "Path does not exist: {path}"
        )));
    }
    if !p.is_dir() {
        return Err(ToolCallResult::error(format!(
            "Path is not a directory: {path}"
        )));
    }
    Ok(p)
}

/// Create a tokio runtime, resolve backend config, and build an embedding provider.
fn create_runtime_and_provider() -> Result<
    (
        tokio::runtime::Runtime,
        vera_core::embedding::DynamicProvider,
        vera_core::config::VeraConfig,
        String,
    ),
    ToolCallResult,
> {
    let backend = vera_core::config::resolve_backend(None);
    let mut config = vera_core::config::VeraConfig::default();
    config.adjust_for_backend(backend);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| ToolCallResult::error(format!("Failed to create runtime: {e}")))?;

    let (provider, model_name) = rt
        .block_on(vera_core::embedding::create_dynamic_provider(
            &config, backend,
        ))
        .map_err(|e| ToolCallResult::error(format!("Failed to create embedding provider: {e}")))?;

    Ok((rt, provider, config, model_name))
}

/// Handle the `index_project` tool.
fn handle_index_project(args: &Value) -> ToolCallResult {
    let repo_path = match require_dir_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let (rt, provider, config, model_name) = match create_runtime_and_provider() {
        Ok(t) => t,
        Err(e) => return e,
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
    let repo_path = match require_dir_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let (rt, provider, config, model_name) = match create_runtime_and_provider() {
        Ok(t) => t,
        Err(e) => return e,
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

/// Handle the `watch_project` tool.
fn handle_watch_project(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };

    match crate::watcher::start_watching(&repo_path) {
        Ok(handle) => {
            let mut guard = WATCHER.lock().unwrap();
            *guard = Some(handle);
            ToolCallResult::success(format!(
                "Watching {} for changes. Index will auto-update when files are modified.",
                repo_path.display()
            ))
        }
        Err(e) => ToolCallResult::error(e),
    }
}

/// Resolve an optional path argument to a validated directory path.
fn resolve_repo_path(args: &Value) -> Result<std::path::PathBuf, ToolCallResult> {
    let repo_path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => std::path::PathBuf::from(p),
        None => std::env::current_dir()
            .map_err(|e| ToolCallResult::error(format!("Failed to get working directory: {e}")))?,
    };
    if !repo_path.exists() {
        return Err(ToolCallResult::error(format!(
            "Path does not exist: {}",
            repo_path.display()
        )));
    }
    Ok(repo_path)
}

/// Handle the `get_stats` tool.
fn handle_get_stats(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match vera_core::stats::collect_stats(&repo_path) {
        Ok(stats) => match serde_json::to_string_pretty(&stats) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize stats: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to collect stats: {e}")),
    }
}

/// Handle the `get_overview` tool.
fn handle_get_overview(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match vera_core::stats::collect_overview(&repo_path) {
        Ok(overview) => match serde_json::to_string_pretty(&overview) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize overview: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to collect overview: {e}")),
    }
}

/// Handle the `find_references` tool.
fn handle_find_references(args: &Value) -> ToolCallResult {
    let symbol = match args.get("symbol").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("Missing required parameter: symbol"),
    };
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("callers");

    match mode {
        "callees" => match vera_core::stats::find_callees(&repo_path, symbol) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(json) => ToolCallResult::success(json),
                Err(e) => ToolCallResult::error(format!("Failed to serialize: {e}")),
            },
            Err(e) => ToolCallResult::error(format!("Failed to find callees: {e}")),
        },
        _ => match vera_core::stats::find_callers(&repo_path, symbol) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(json) => ToolCallResult::success(json),
                Err(e) => ToolCallResult::error(format!("Failed to serialize: {e}")),
            },
            Err(e) => ToolCallResult::error(format!("Failed to find callers: {e}")),
        },
    }
}

/// Handle the `find_dead_code` tool.
fn handle_find_dead_code(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match vera_core::stats::find_dead_symbols(&repo_path) {
        Ok(results) => match serde_json::to_string(&results) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to find dead code: {e}")),
    }
}

/// Handle the `regex_search` tool.
fn handle_regex_search(args: &Value) -> ToolCallResult {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolCallResult::error("Missing required parameter: pattern"),
    };
    let scope = match args.get("scope").and_then(|v| v.as_str()) {
        Some(value) => match value.parse() {
            Ok(scope) => Some(scope),
            Err(()) => {
                return ToolCallResult::error(format!("Invalid scope: {value}"));
            }
        },
        None => None,
    };

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(20);
    let ignore_case = args
        .get("ignore_case")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let context = args
        .get("context")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(2);

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

    let filters = vera_core::types::SearchFilters {
        scope,
        include_generated: Some(
            args.get("include_generated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        ),
        ..Default::default()
    };

    match vera_core::retrieval::search_regex(
        &index_dir,
        pattern,
        limit,
        ignore_case,
        context,
        &filters,
    ) {
        Ok(results) => match compact_results_json(&results) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Regex search failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_has_nine_tools() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 9);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"index_project"));
        assert!(names.contains(&"update_project"));
        assert!(names.contains(&"get_stats"));
        assert!(names.contains(&"get_overview"));
        assert!(names.contains(&"watch_project"));
        assert!(names.contains(&"find_references"));
        assert!(names.contains(&"find_dead_code"));
        assert!(names.contains(&"regex_search"));
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
