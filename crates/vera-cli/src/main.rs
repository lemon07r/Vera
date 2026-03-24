//! Vera CLI — code indexing and retrieval for local and tool-driven workflows.
//!
//! # Commands
//!
//! - `vera index <path>` — Index a codebase for search
//! - `vera search <query>` — Search the indexed codebase
//! - `vera update <path>` — Incrementally update the index
//! - `vera stats` — Show index statistics
//! - `vera config` — Show or set configuration values

mod commands;
mod helpers;

use std::process;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vera",
    about = "Hybrid code indexing and retrieval for local and tool-driven workflows",
    long_about = "Vera is a code indexing and retrieval tool for source trees. It combines \
                  BM25 full-text search with vector similarity search using Reciprocal Rank \
                  Fusion (RRF) and optional cross-encoder reranking to return ranked code \
                  results for direct CLI use and tool integrations.\n\n\
                  Quick start:\n  \
                  vera index .          # Index current directory\n  \
                  vera search \"auth\"    # Search for authentication code\n  \
                  vera update .         # Update index after changes\n  \
                  vera stats            # Show index statistics",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output results as JSON (machine-readable).
    ///
    /// When enabled, all data output goes to stdout as valid JSON.
    /// Logs and diagnostics always go to stderr regardless of this flag.
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP (Model Context Protocol) server.
    ///
    /// Runs a JSON-RPC 2.0 server over stdio for tool integration.
    /// The server exposes tools: search_code, index_project, update_project, get_stats.
    ///
    /// Examples:
    ///   vera mcp
    #[command(long_about = "Start the MCP (Model Context Protocol) server.\n\n\
                      Runs a JSON-RPC 2.0 server over stdio so editors, assistants, and \
                      other tools can use Vera's indexing and search capabilities.\n\n\
                      The server reads JSON-RPC messages from stdin and writes responses \
                      to stdout. Logs go to stderr.\n\n\
                      Exposed tools:\n  \
                      search_code     — Hybrid search with filters\n  \
                      index_project   — Index a project directory\n  \
                      update_project  — Incremental index update\n  \
                      get_stats       — Index statistics\n\n\
                      Examples:\n  \
                      vera mcp                       # Start MCP server on stdio")]
    Mcp,

    /// Index a codebase for search.
    ///
    /// Discovers source files, parses them with tree-sitter, creates
    /// searchable chunks, generates embeddings, and stores everything
    /// in a local `.vera/` index directory.
    ///
    /// Examples:
    ///   vera index .
    ///   vera index /path/to/repo
    ///   vera index . --json
    #[command(long_about = "Index a codebase for search.\n\n\
                      Discovers source files (respecting .gitignore), parses them with \
                      tree-sitter for 20+ languages, creates searchable chunks at symbol \
                      boundaries, generates embeddings via API, and stores everything in \
                      a local `.vera/` index directory.\n\n\
                      Requires EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and \
                      EMBEDDING_MODEL_API_KEY environment variables.\n\n\
                      Examples:\n  \
                      vera index .                  # Index current directory\n  \
                      vera index /path/to/repo      # Index a specific repo\n  \
                      vera index . --json           # Output summary as JSON")]
    Index {
        /// Path to the directory to index.
        path: String,
        /// Use local inference for embedding (overrides API provider).
        #[arg(long)]
        local: bool,
    },

    /// Search the indexed codebase.
    ///
    /// Performs hybrid search combining BM25 keyword matching and vector
    /// similarity, fused with Reciprocal Rank Fusion (RRF). Optional
    /// cross-encoder reranking for improved precision.
    ///
    /// Examples:
    ///   vera search "authentication logic"
    ///   vera search "parse_config" --lang rust
    ///   vera search "error handling" --limit 5 --json
    #[command(long_about = "Search the indexed codebase.\n\n\
                      Performs hybrid search combining BM25 keyword matching and vector \
                      similarity via Reciprocal Rank Fusion (RRF). Optional cross-encoder \
                      reranking for improved precision.\n\n\
                      Falls back gracefully: if embedding API is unavailable, uses BM25-only \
                      search. If reranker is unavailable, returns unreranked hybrid results.\n\n\
                      Requires an existing index (run `vera index <path>` first).\n\n\
                      Examples:\n  \
                      vera search \"auth logic\"                  # Semantic search\n  \
                      vera search \"parse_config\"                 # Symbol lookup\n  \
                      vera search \"error handling\" --lang rust   # Filter by language\n  \
                      vera search \"routes\" --path \"src/**/*.ts\"  # Filter by path\n  \
                      vera search \"DB queries\" --type function   # Filter by symbol type\n  \
                      vera search \"config\" --limit 5 --json      # JSON output, 5 results")]
    Search {
        /// The search query (keyword or natural language).
        query: String,

        /// Filter by programming language (case-insensitive).
        ///
        /// Restricts results to the specified language.
        /// Supported: rust, typescript, python, go, java, c, cpp, etc.
        #[arg(long)]
        lang: Option<String>,

        /// Filter by file path glob pattern (e.g., "src/**/*.rs").
        ///
        /// Supports * (any within segment) and ** (any depth).
        #[arg(long)]
        path: Option<String>,

        /// Maximum number of results to return (default: 10).
        #[arg(long, short = 'n')]
        limit: Option<usize>,

        /// Filter by symbol type.
        ///
        /// Options: function, method, class, struct, enum, trait,
        /// interface, type_alias, constant, variable, module, block.
        #[arg(long, rename_all = "snake_case")]
        r#type: Option<String>,

        /// Use local inference for embedding and reranking.
        #[arg(long)]
        local: bool,
    },

    /// Incrementally update the index for changed files.
    ///
    /// Detects files that have been added, modified, or deleted since
    /// the last index/update, and only re-processes changed files.
    /// Much faster than a full re-index.
    ///
    /// Examples:
    ///   vera update .
    ///   vera update /path/to/repo --json
    #[command(long_about = "Incrementally update the index for changed files.\n\n\
                      Uses content hashing to detect files that have been added, modified, \
                      or deleted since the last index/update. Only changed files are \
                      re-processed, making updates much faster than a full re-index.\n\n\
                      Requires EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and \
                      EMBEDDING_MODEL_API_KEY environment variables.\n\n\
                      Examples:\n  \
                      vera update .                  # Update current directory\n  \
                      vera update /path/to/repo      # Update a specific repo\n  \
                      vera update . --json           # Output summary as JSON")]
    Update {
        /// Path to the directory to update.
        path: String,
        /// Use local inference for embedding (overrides API provider).
        #[arg(long)]
        local: bool,
    },

    /// Show index statistics.
    ///
    /// Displays file count, chunk count, index size on disk,
    /// and language breakdown for the current index.
    ///
    /// Examples:
    ///   vera stats
    ///   vera stats --json
    #[command(long_about = "Show index statistics.\n\n\
                      Displays file count, chunk count, index size on disk, and a breakdown \
                      of chunks by programming language for the current index.\n\n\
                      Looks for the index in the current working directory (`.vera/`).\n\n\
                      Examples:\n  \
                      vera stats             # Human-readable stats\n  \
                      vera stats --json      # Machine-readable JSON output")]
    Stats,

    /// Show or set configuration values.
    ///
    /// Without arguments, shows the current configuration. Use subcommands
    /// to get or set individual values.
    ///
    /// Examples:
    ///   vera config
    ///   vera config get retrieval.default_limit
    ///   vera config set retrieval.default_limit 20
    #[command(long_about = "Show or set configuration values.\n\n\
                      Without arguments or with `show`, displays the full current \
                      configuration as a table (or JSON with --json).\n\n\
                      Use `get <key>` to read a specific value, or `set <key> <value>` \
                      to update it.\n\n\
                      Configuration keys use dot notation:\n  \
                      indexing.max_chunk_lines       Max lines per chunk (default: 200)\n  \
                      indexing.max_file_size_bytes   Max file size to index (default: 1000000)\n  \
                      retrieval.default_limit        Default result count (default: 10)\n  \
                      retrieval.rrf_k                RRF fusion constant (default: 60)\n  \
                      retrieval.rerank_candidates    Reranker candidate count (default: 50)\n  \
                      retrieval.reranking_enabled    Enable reranking (default: true)\n  \
                      embedding.batch_size           Embedding batch size (default: 128)\n  \
                      embedding.max_concurrent_requests  Concurrent API requests (default: 8)\n  \
                      embedding.timeout_secs         API timeout (default: 60)\n  \
                      embedding.max_retries          API retry count (default: 3)\n  \
                      embedding.max_stored_dim       Vector dimensionality (default: 1024)\n\n\
                      Examples:\n  \
                      vera config                                  # Show all settings\n  \
                      vera config show                             # Same as above\n  \
                      vera config get retrieval.default_limit      # Get one value\n  \
                      vera config set retrieval.default_limit 20   # Set a value\n  \
                      vera config --json                           # JSON output")]
    Config {
        /// Config action: show (default), get <key>, or set <key> <value>.
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}

fn main() {
    // Initialize tracing subscriber (logs go to stderr).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("VERA_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Mcp => {
            tracing::info!("starting MCP server");
            commands::mcp::run();
            Ok(())
        }
        Commands::Index { path, local } => {
            tracing::info!(path = %path, "indexing");
            commands::index::run(&path, cli.json, local)
        }
        Commands::Search {
            query,
            lang,
            path,
            limit,
            r#type,
            local,
        } => {
            tracing::info!(query = %query, "searching");
            let filters = vera_core::types::SearchFilters {
                language: lang,
                path_glob: path,
                symbol_type: r#type,
            };
            commands::search::run(&query, limit, &filters, cli.json, local)
        }
        Commands::Update { path, local } => {
            tracing::info!(path = %path, "updating");
            commands::update::run(&path, cli.json, local)
        }
        Commands::Stats => {
            tracing::info!("showing stats");
            commands::stats::run(cli.json)
        }
        Commands::Config { args } => {
            tracing::info!("config command");
            commands::config::run(&args, cli.json)
        }
    };

    if let Err(err) = result {
        eprintln!("Error: {err:#}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_index_command() {
        let cli = Cli::parse_from(["vera", "index", "/tmp/repo"]);
        assert!(matches!(cli.command, Commands::Index { path, .. } if path == "/tmp/repo"));
    }

    #[test]
    fn cli_parses_search_command() {
        let cli = Cli::parse_from(["vera", "search", "find auth"]);
        assert!(matches!(cli.command, Commands::Search { query, .. } if query == "find auth"));
    }

    #[test]
    fn cli_parses_search_with_filters() {
        let cli = Cli::parse_from([
            "vera",
            "search",
            "find auth",
            "--lang",
            "rust",
            "--limit",
            "5",
        ]);
        match cli.command {
            Commands::Search {
                query, lang, limit, ..
            } => {
                assert_eq!(query, "find auth");
                assert_eq!(lang, Some("rust".to_string()));
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_type_filter() {
        let cli = Cli::parse_from(["vera", "search", "find auth", "--type", "function"]);
        match cli.command {
            Commands::Search { query, r#type, .. } => {
                assert_eq!(query, "find auth");
                assert_eq!(r#type, Some("function".to_string()));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_path_filter() {
        let cli = Cli::parse_from(["vera", "search", "config", "--path", "src/**/*.rs"]);
        match cli.command {
            Commands::Search { query, path, .. } => {
                assert_eq!(query, "config");
                assert_eq!(path, Some("src/**/*.rs".to_string()));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_all_filters() {
        let cli = Cli::parse_from([
            "vera",
            "search",
            "handle request",
            "--lang",
            "typescript",
            "--path",
            "src/**/*.ts",
            "--type",
            "function",
            "--limit",
            "3",
        ]);
        match cli.command {
            Commands::Search {
                query,
                lang,
                path,
                limit,
                r#type,
                ..
            } => {
                assert_eq!(query, "handle request");
                assert_eq!(lang, Some("typescript".to_string()));
                assert_eq!(path, Some("src/**/*.ts".to_string()));
                assert_eq!(r#type, Some("function".to_string()));
                assert_eq!(limit, Some(3));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_update_command() {
        let cli = Cli::parse_from(["vera", "update", "/tmp/repo"]);
        assert!(matches!(cli.command, Commands::Update { path, .. } if path == "/tmp/repo"));
    }

    #[test]
    fn cli_parses_stats_command() {
        let cli = Cli::parse_from(["vera", "stats"]);
        assert!(matches!(cli.command, Commands::Stats));
    }

    #[test]
    fn cli_parses_json_flag() {
        let cli = Cli::parse_from(["vera", "--json", "stats"]);
        assert!(cli.json);
    }

    #[test]
    fn cli_parses_config_command() {
        let cli = Cli::parse_from(["vera", "config"]);
        assert!(matches!(cli.command, Commands::Config { args } if args.is_empty()));
    }

    #[test]
    fn cli_parses_config_show() {
        let cli = Cli::parse_from(["vera", "config", "show"]);
        match cli.command {
            Commands::Config { args } => {
                assert_eq!(args, vec!["show".to_string()]);
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_get() {
        let cli = Cli::parse_from(["vera", "config", "get", "retrieval.default_limit"]);
        match cli.command {
            Commands::Config { args } => {
                assert_eq!(
                    args,
                    vec!["get".to_string(), "retrieval.default_limit".to_string()]
                );
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_set() {
        let cli = Cli::parse_from(["vera", "config", "set", "retrieval.default_limit", "20"]);
        match cli.command {
            Commands::Config { args } => {
                assert_eq!(
                    args,
                    vec![
                        "set".to_string(),
                        "retrieval.default_limit".to_string(),
                        "20".to_string()
                    ]
                );
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn config_get_known_keys() {
        let config = vera_core::config::VeraConfig::default();
        assert!(commands::config::get_config_value(&config, "indexing.max_chunk_lines").is_some());
        assert!(commands::config::get_config_value(&config, "retrieval.default_limit").is_some());
        assert!(commands::config::get_config_value(&config, "retrieval.rrf_k").is_some());
        assert!(
            commands::config::get_config_value(&config, "retrieval.reranking_enabled").is_some()
        );
        assert!(commands::config::get_config_value(&config, "embedding.batch_size").is_some());
        assert!(commands::config::get_config_value(&config, "embedding.max_stored_dim").is_some());
    }

    #[test]
    fn config_get_unknown_key_returns_none() {
        let config = vera_core::config::VeraConfig::default();
        assert!(commands::config::get_config_value(&config, "nonexistent.key").is_none());
    }

    #[test]
    fn config_values_match_defaults() {
        let config = vera_core::config::VeraConfig::default();
        let val = commands::config::get_config_value(&config, "retrieval.default_limit").unwrap();
        assert_eq!(val, serde_json::json!(10));

        let val = commands::config::get_config_value(&config, "indexing.max_chunk_lines").unwrap();
        assert_eq!(val, serde_json::json!(200));

        let val =
            commands::config::get_config_value(&config, "retrieval.reranking_enabled").unwrap();
        assert_eq!(val, serde_json::json!(true));
    }
}
