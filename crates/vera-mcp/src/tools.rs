//! MCP tool definitions and handler dispatch.
//!
//! Defines the tools that the Vera MCP server exposes:
//! - `search_code` — search indexed codebase (auto-indexes and watches on first use)
//! - `get_stats` — retrieve index statistics
//! - `get_overview` — architecture overview for agent onboarding
//! - `regex_search` — regex search over indexed files

use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;

use crate::protocol::{ToolCallResult, ToolDefinition};
use crate::watcher::WatchHandle;

/// Global watcher handle. Kept alive for the lifetime of the MCP server process.
static WATCHER: Mutex<Option<WatchHandle>> = Mutex::new(None);

/// Default total output budget for MCP responses (chars).
const MCP_OUTPUT_BUDGET: usize = 20_000;

/// Compact result representation for MCP tool responses.
/// Drops `score` and `language` (inferrable from extension), omits null fields.
#[derive(Serialize)]
struct CompactResult<'a> {
    file_path: &'a str,
    line_start: u32,
    line_end: u32,
    content: std::borrow::Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_type: Option<&'a vera_core::types::SymbolType>,
}

/// Truncate `content` to fit within `allowed` bytes, breaking at a line boundary.
fn truncate_to_budget(content: &str, allowed: usize) -> std::borrow::Cow<'_, str> {
    if content.len() <= allowed {
        return std::borrow::Cow::Borrowed(content);
    }
    let end = content
        .char_indices()
        .take_while(|(i, _)| *i < allowed)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let break_at = content[..end].rfind('\n').unwrap_or(end);
    let mut truncated = content[..break_at].to_string();
    truncated.push_str("\n[...truncated]");
    std::borrow::Cow::Owned(truncated)
}

/// Serialize search results as compact JSON, applying a total character budget.
fn compact_results_json(
    results: &[vera_core::types::SearchResult],
    budget: usize,
) -> Result<String, serde_json::Error> {
    let mut remaining = budget;
    let mut compact: Vec<CompactResult> = Vec::with_capacity(results.len());
    for r in results {
        if budget > 0 && remaining == 0 {
            break;
        }
        let content = if budget > 0 {
            let c = truncate_to_budget(&r.content, remaining);
            remaining = remaining.saturating_sub(c.len());
            c
        } else {
            std::borrow::Cow::Borrowed(r.content.as_str())
        };
        compact.push(CompactResult {
            file_path: &r.file_path,
            line_start: r.line_start,
            line_end: r.line_end,
            content,
            symbol_name: r.symbol_name.as_deref(),
            symbol_type: r.symbol_type.as_ref(),
        });
    }
    serde_json::to_string(&compact)
}

/// Return the list of tools the server advertises.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_code".to_string(),
            description: "Search the indexed codebase using hybrid BM25+vector \
                          retrieval with cross-encoder reranking. Returns ranked \
                          code snippets with file paths, line numbers, and content.\n\
                          \n\
                          WHEN TO USE: conceptual or behavioral queries (\"how is auth handled\", \
                          \"error retry logic\", \"database connection pooling\"). Understands \
                          synonyms and related concepts.\n\
                          WHEN NOT TO USE: exact string matching, regex patterns, or \
                          import statements. Use regex_search for those.\n\
                          \n\
                          TIPS: Use 2-3 varied queries to capture different aspects of what \
                          you are looking for. Set intent to describe your higher-level goal \
                          for better reranking."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keyword or natural language)"
                    },
                    "queries": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Multiple search queries to run in parallel and merge. Use 2-3 varied queries to capture different aspects (e.g., ['OAuth token refresh', 'JWT expiry handling', 'auth middleware']). Results are deduplicated and reranked."
                    },
                    "intent": {
                        "type": "string",
                        "description": "Higher-level goal for reranking (e.g., 'find where auth tokens are validated and refreshed'). Improves precision when the query is ambiguous."
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
                        "description": "Maximum number of results (default: 5)"
                    }
                },
                "required": []
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
                          directories, entry points, symbol types, complexity hotspots, \
                          and detected project conventions (frameworks, patterns, config files). \
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
            name: "regex_search".to_string(),
            description: "Search indexed files using a regex pattern. Returns matches \
                          with surrounding context lines.\n\
                          \n\
                          WHEN TO USE: exact string matching, regex patterns, import statements, \
                          TODOs, specific syntax, or known identifiers.\n\
                          WHEN NOT TO USE: conceptual or behavioral queries. Use search_code \
                          for those."
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
        "get_stats" => handle_get_stats(arguments),
        "get_overview" => handle_get_overview(arguments),
        "regex_search" => handle_regex_search(arguments),
        _ => ToolCallResult::error(format!("Unknown tool: {name}")),
    }
}

/// Handle the `search_code` tool.
fn handle_search_code(args: &Value) -> ToolCallResult {
    // Collect queries: support both single `query` and multi `queries`.
    let mut queries: Vec<String> = Vec::new();
    if let Some(q) = args.get("query").and_then(|v| v.as_str()) {
        queries.push(q.to_string());
    }
    if let Some(arr) = args.get("queries").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(q) = item.as_str() {
                queries.push(q.to_string());
            }
        }
    }
    if queries.is_empty() {
        return ToolCallResult::error(
            "Missing required parameter: provide 'query' (string) or 'queries' (array)",
        );
    }

    let intent = args.get("intent").and_then(|v| v.as_str());

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

    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => return ToolCallResult::error(format!("Failed to get working directory: {e}")),
    };
    let index_dir = vera_core::indexing::index_dir(&cwd);

    if !index_dir.exists() {
        // Auto-index on first search.
        let (rt, provider, idx_config, model_name) = match create_runtime_and_provider() {
            Ok(t) => t,
            Err(e) => return e,
        };
        match rt.block_on(vera_core::indexing::index_repository(
            &cwd,
            &provider,
            &idx_config,
            &model_name,
        )) {
            Ok(_) => {}
            Err(e) => return ToolCallResult::error(format!("Auto-indexing failed: {e}")),
        }
        // Start watcher after indexing.
        if let Ok(handle) = crate::watcher::start_watching(&cwd) {
            let mut guard = WATCHER.lock().unwrap();
            *guard = Some(handle);
        }
    } else {
        // Start watcher if not already running.
        let mut guard = WATCHER.lock().unwrap();
        if guard.is_none() {
            if let Ok(handle) = crate::watcher::start_watching(&cwd) {
                *guard = Some(handle);
            }
        }
    }

    // Run each query, collect all results.
    let mut all_results: Vec<vera_core::types::SearchResult> = Vec::new();
    let per_query_limit = if queries.len() > 1 {
        result_limit.max(10)
    } else {
        result_limit
    };

    for query in &queries {
        // If intent is provided, prepend it to the query for better reranking.
        let effective_query = match intent {
            Some(i) => format!("intent: {i} | {query}"),
            None => query.clone(),
        };
        match vera_core::retrieval::search_service::execute_search(
            &index_dir,
            &effective_query,
            &config,
            &filters,
            per_query_limit,
            backend,
        ) {
            Ok((results, _timings)) => all_results.extend(results),
            Err(e) => return ToolCallResult::error(format!("Search failed: {e}")),
        }
    }

    // Deduplicate by (file_path, line_start, line_end), keeping first occurrence.
    let mut seen = std::collections::HashSet::new();
    all_results.retain(|r| seen.insert(format!("{}:{}:{}", r.file_path, r.line_start, r.line_end)));
    all_results.truncate(result_limit);

    match compact_results_json(&all_results, MCP_OUTPUT_BUDGET) {
        Ok(json) => ToolCallResult::success(json),
        Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
    }
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
            "No index found in current directory. Run search_code first to auto-index.",
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
        Ok(results) => match compact_results_json(&results, MCP_OUTPUT_BUDGET) {
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
    fn tool_definitions_has_four_tools() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"get_stats"));
        assert!(names.contains(&"get_overview"));
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
    fn search_code_accepts_queries_array() {
        // No index and no embedding config, should get past parameter validation.
        let result = handle_tool_call(
            "search_code",
            &serde_json::json!({"queries": ["foo", "bar"]}),
        );
        // Should fail (either auto-index fails or embedding provider fails).
        assert!(result.is_error);
    }

    #[test]
    fn truncate_to_budget_short_passthrough() {
        let short = "hello world";
        let result = truncate_to_budget(short, 1000);
        assert_eq!(result.as_ref(), short);
    }

    #[test]
    fn truncate_to_budget_long_truncates() {
        let long = "a".repeat(500);
        let result = truncate_to_budget(&long, 100);
        assert!(result.len() < long.len());
        assert!(result.ends_with("[...truncated]"));
    }

    #[test]
    fn removed_tools_return_unknown() {
        for tool in &[
            "index_project",
            "update_project",
            "watch_project",
            "find_references",
            "find_dead_code",
        ] {
            let result = handle_tool_call(tool, &serde_json::json!({}));
            assert!(result.is_error);
            assert!(result.content[0].text.contains("Unknown tool"));
        }
    }

    #[test]
    fn get_stats_no_index_returns_error() {
        let result = handle_tool_call("get_stats", &serde_json::json!({"path": "/tmp"}));
        assert!(result.is_error);
        // Should mention no index found or similar.
        assert!(!result.content[0].text.is_empty());
    }
}
