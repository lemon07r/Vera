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
mod skill_assets;
mod state;
mod update_check;

use std::process;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vera",
    about = "Hybrid code indexing and retrieval for CLI-first coding-agent workflows",
    long_about = "Vera is a code indexing and retrieval tool for source trees. It combines \
                  BM25 full-text search with vector similarity search using Reciprocal Rank \
                  Fusion (RRF) and optional cross-encoder reranking to return ranked code \
                  results for direct CLI use and installable agent skills. Vera always keeps \
                  the index local in `.vera/`; `vera setup` only chooses the model backend.\n\n\
                  Quick start:\n  \
                  vera agent install                   # Interactive: choose scope + agents\n  \
                  vera setup                          # Download built-in local models\n  \
                  vera index .                        # Index current directory\n  \
                  vera search \"auth\"                  # Search for authentication code\n  \
                  vera doctor                         # Check local setup and index health\n  \
                  vera repair                         # Re-fetch missing backend assets\n  \
                  vera upgrade                        # Show the binary update plan",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output results as compact JSON (machine-readable).
    ///
    /// By default, search output uses markdown codeblocks optimized for
    /// AI agent token budgets. Use --json for compact single-line JSON
    /// (useful for programmatic consumption or piping to other tools).
    #[arg(long, global = true)]
    json: bool,

    /// Output all fields with pretty-printed verbose formatting.
    ///
    /// Shows numbered results with scores, language, symbol info, and
    /// decorated code previews. Useful for human debugging.
    #[arg(long, global = true)]
    raw: bool,

    /// Print search pipeline step timings to stderr.
    #[arg(long, global = true)]
    timing: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP (Model Context Protocol) server.
    ///
    /// Runs a JSON-RPC 2.0 server over stdio for tool integration.
    /// The server exposes tools: search_code, index_project, update_project, get_stats,
    /// get_overview, watch_project, find_references, find_dead_code, and regex_search.
    ///
    /// Examples:
    ///   vera mcp
    #[command(long_about = "Start the MCP (Model Context Protocol) server.\n\n\
                      Runs a JSON-RPC 2.0 server over stdio so editors, assistants, and \
                      other tools can use Vera's indexing and search capabilities.\n\n\
                      The server reads JSON-RPC messages from stdin and writes responses \
                      to stdout. Logs go to stderr.\n\n\
                      Exposed tools:\n  \
                      search_code      — Hybrid search with filters\n  \
                      index_project    — Index a project directory\n  \
                      update_project   — Incremental index update\n  \
                      get_stats        — Index statistics\n  \
                      get_overview     — Project summary for onboarding\n  \
                      watch_project    — Watch files and auto-update the index\n  \
                      find_references  — Find callers or callees of a symbol\n  \
                      find_dead_code   — Find functions with no callers\n  \
                      regex_search     — Regex search over indexed files\n\n\
                      Examples:\n  \
                      vera mcp                       # Start MCP server on stdio")]
    Mcp,

    /// Install or manage the Vera skill for supported coding agents.
    ///
    /// This is the preferred agent integration path. It writes the canonical
    /// `skills/vera` bundle into well-known skill directories for supported
    /// clients, so agents can use the Vera CLI directly without MCP.
    #[command(
        long_about = "Install or manage the Vera skill for supported coding agents.\n\n\
                      This is the preferred agent integration path. Vera installs a \
                      CLI-centric skill bundle into known skill directories so agents \
                      can call `vera index`, `vera search`, `vera update`, and \
                      `vera stats` directly.\n\n\
                      `vera agent install` is idempotent: rerun it to refresh an \
                      existing skill install.\n\n\
                      Examples:\n  \
                      vera agent install                       # Interactive: choose scope and agents\n  \
                      vera agent install --client claude       # Install for Claude Code (global)\n  \
                      vera agent install --client all --scope project  # All agents, project only\n  \
                      vera agent status                        # Show all install status\n  \
                      vera agent remove                        # Interactive: pick installs to remove\n  \
                      vera agent remove --client codex         # Remove the global Codex install"
    )]
    Agent {
        /// Agent command: install, status, or remove.
        #[arg(value_enum)]
        command: commands::agent::AgentCommand,
        /// Which agent client to target. Without this flag, interactive mode
        /// presents a checklist of all supported agents.
        #[arg(long, value_enum)]
        client: Option<commands::agent::AgentClient>,
        /// Install scope: global, project, or all. Without this flag,
        /// interactive mode prompts for scope selection.
        #[arg(long, value_enum)]
        scope: Option<commands::agent::AgentScope>,
    },

    /// Remove Vera: binary, models, config, agent skills, and PATH shim.
    ///
    /// Per-project indexes (.vera/) are not removed.
    #[command(
        long_about = "Remove Vera: binary cache, models, ONNX Runtime libs, config, \n\
                      credentials, agent skill files, and the PATH shim.\n\n\
                      Per-project indexes (.vera/ inside each project) are not touched.\n\n\
                      Examples:\n  \
                      vera uninstall\n  \
                      vera uninstall --json"
    )]
    Uninstall,

    /// Persist a preferred model backend and bootstrap first-run state.
    ///
    /// By default this downloads the built-in local model assets and optionally
    /// indexes a repository immediately.
    #[command(
        long_about = "Persist a preferred model backend and bootstrap first-run state.\n\n\
                      Running `vera setup` with no flags shows an interactive backend menu \
                      with auto-detected GPU as the default. Pass a --onnx-jina-* flag or \
                      --api to skip the menu. Add `--code-rank-embed` to use the optional \
                      CodeRankEmbed local embedding model, or `--embedding-repo/--embedding-dir` \
                      for a custom ONNX embedding model.\n\n\
                      Vera always keeps the index local in `.vera/`. The choice here only \
                      changes where embeddings and reranking are computed.\n\n\
                      Examples:\n  \
                      vera setup                       # Interactive backend selection\n  \
                      vera setup --onnx-jina-cuda      # NVIDIA GPU (skip menu)\n  \
                      vera setup --code-rank-embed     # Switch local embeddings to CodeRankEmbed\n  \
                      vera setup --embedding-repo <repo-or-url> --embedding-pooling cls\n  \
                      vera setup --api                 # Persist API credentials from env\n  \
                      vera setup --yes                 # Auto-detect GPU, no prompts"
    )]
    Setup {
        #[command(flatten)]
        backend: helpers::LocalBackendFlags,
        #[command(flatten)]
        embedding: helpers::LocalEmbeddingModelFlags,
        /// Configure Vera for API-backed mode using current env vars.
        #[arg(long, group = "backend")]
        api: bool,
        /// Optionally index a repository after saving config.
        #[arg(long)]
        index: Option<String>,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },

    /// Inspect the current Vera setup for common configuration issues.
    ///
    /// Checks the persisted config, effective mode, local runtime or API env,
    /// and whether the current repository has an index.
    #[command(
        long_about = "Inspect the current Vera setup for common configuration issues.\n\n\
                      Checks the persisted config, effective mode, local runtime or \
                      API environment variables, and whether the current repository \
                      has a `.vera/` index. `--probe` adds a deeper read-only ONNX \
                      session probe and never downloads or repairs missing assets.\n\n\
                      Examples:\n  \
                      vera doctor\n  \
                      vera doctor --probe\n  \
                      vera doctor --json"
    )]
    Doctor {
        /// Run a deeper read-only probe of local ONNX session init.
        #[arg(long, visible_alias = "deep")]
        probe: bool,
    },

    /// Repair the configured Vera backend.
    ///
    /// Re-fetches missing local runtime/model assets for the selected local
    /// backend, or re-persists API configuration from the current environment.
    #[command(long_about = "Repair the configured Vera backend.\n\n\
                      For local ONNX backends, this re-fetches missing runtime and \
                      model assets for the selected backend. For API mode, it re-saves \
                      the current API environment variables into Vera's config.\n\n\
                      This is a write operation. Use `vera doctor --probe` for a read-only \
                      diagnostic check.\n\n\
                      Examples:\n  \
                      vera repair\n  \
                      vera repair --onnx-jina-cuda\n  \
                      vera repair --api")]
    Repair {
        #[command(flatten)]
        backend: helpers::LocalBackendFlags,
        /// Repair API-backed mode using current env vars.
        #[arg(long, group = "backend")]
        api: bool,
    },

    /// Show the binary update plan, or apply it when the install method is known.
    #[command(
        long_about = "Show the binary update plan, or apply it when the install method is known.\n\n\
                      By default, `vera upgrade` is a dry run: it checks for a newer Vera \
                      release, resolves the saved or detected install method, and prints the \
                      exact command it would use.\n\n\
                      `--apply` runs the installer command only when Vera can determine a \
                      single install method. If multiple install methods are detected, Vera \
                      prints the manual options and refuses to guess.\n\n\
                      Examples:\n  \
                      vera upgrade\n  \
                      vera upgrade --apply\n  \
                      vera upgrade --json"
    )]
    Upgrade {
        /// Run the planned installer command instead of printing it only.
        #[arg(long)]
        apply: bool,
    },

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
                      tree-sitter for 60+ languages, creates searchable chunks at symbol \
                      boundaries, generates embeddings using the current Vera mode, and \
                      stores everything in a local `.vera/` index directory.\n\n\
                      Use `vera setup` for Vera's built-in local models, or `vera setup \
                      --api` for an OpenAI-compatible endpoint.\n\n\
                      Examples:\n  \
                      vera index .                  # Index current directory\n  \
                      vera index /path/to/repo      # Index a specific repo\n  \
                      vera index . --json           # Output summary as JSON")]
    Index {
        /// Path to the directory to index.
        path: String,
        #[command(flatten)]
        backend: helpers::LocalBackendFlags,
        /// Exclude files matching this glob pattern (repeatable).
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Disable .gitignore and .veraignore parsing.
        #[arg(long)]
        no_ignore: bool,
        /// Disable smart default exclusions.
        #[arg(long)]
        no_default_excludes: bool,
        /// Show detailed information (e.g. paths of skipped files).
        #[arg(long, short = 'v')]
        verbose: bool,
        /// Reduce GPU memory usage (batch_size=1, conservative VRAM limit).
        #[arg(long)]
        low_vram: bool,
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
                      Source files are favored by default. Use `--scope docs` for prose, \
                      `--scope runtime` for extracted runtime trees, and `--include-generated` \
                      when you intentionally want dist/minified artifacts.\n\n\
                      Falls back gracefully: if embedding API is unavailable, uses BM25-only \
                      search. If reranker is unavailable, returns unreranked hybrid results.\n\n\
                      Requires an existing index (run `vera index <path>` first).\n\n\
                      Examples:\n  \
                      vera search \"auth logic\"                  # Semantic search\n  \
                      vera search \"parse_config\"                 # Symbol lookup\n  \
                      vera search \"hotkeys\" --scope docs         # Search docs only\n  \
                      vera search \"mod loader\" --scope runtime --include-generated\n  \
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

        /// Restrict results to a coarse corpus scope.
        #[arg(long, value_parser = ["source", "docs", "runtime", "all"])]
        scope: Option<String>,

        /// Include generated or minified files such as dist bundles.
        #[arg(long)]
        include_generated: bool,

        /// Multi-hop iterative search: follow up on symbols found in initial results.
        #[arg(long)]
        deep: bool,

        #[command(flatten)]
        backend: helpers::LocalBackendFlags,
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
                      Uses the saved Vera mode from `vera setup`, or the current shell \
                      environment if you are configuring providers manually.\n\n\
                      Examples:\n  \
                      vera update .                  # Update current directory\n  \
                      vera update /path/to/repo      # Update a specific repo\n  \
                      vera update . --json           # Output summary as JSON")]
    Update {
        /// Path to the directory to update.
        path: String,
        #[command(flatten)]
        backend: helpers::LocalBackendFlags,
        /// Exclude files matching this glob pattern (repeatable).
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Disable .gitignore and .veraignore parsing.
        #[arg(long)]
        no_ignore: bool,
        /// Disable smart default exclusions.
        #[arg(long)]
        no_default_excludes: bool,
    },

    /// Show architecture overview of the indexed project.
    ///
    /// Returns a high-level summary: languages, directories, entry points,
    /// symbol types, and complexity hotspots. Useful for onboarding.
    ///
    /// Examples:
    ///   vera overview
    ///   vera overview --json
    #[command(long_about = "Show architecture overview of the indexed project.\n\n\
                      Returns a high-level summary of the codebase: languages with file \n\
                      and chunk counts, top-level directories, symbol type breakdown, \n\
                      likely entry points, and complexity hotspots.\n\n\
                      Useful for quick orientation when starting work on a new project.\n\n\
                      Examples:\n  \
                      vera overview             # Human-readable overview\n  \
                      vera overview --json      # Machine-readable JSON output")]
    Overview,

    /// Find callers or callees of a symbol.
    ///
    /// Queries the call graph built during indexing to find where a symbol
    /// is called from (callers) or what it calls (callees).
    ///
    /// Examples:
    ///   vera references parse_and_chunk
    ///   vera references parse_and_chunk --callees
    ///   vera references parse_and_chunk --json
    References {
        /// Symbol name to look up.
        symbol: String,
        /// Show what this symbol calls instead of what calls it.
        #[arg(long)]
        callees: bool,
    },

    /// Regex pattern search over indexed files.
    ///
    /// Searches file contents using a regex pattern, returning matches
    /// with surrounding context. Only searches files in the index.
    ///
    /// Examples:
    ///   vera grep "fn\s+main"
    ///   vera grep "TODO|FIXME" -i
    ///   vera grep "impl.*Display" --context 5
    #[command(long_about = "Regex pattern search over indexed files.\n\n\
                      Searches file contents using a regex pattern, returning matches \
                      with surrounding context lines. Only searches files that are in \
                      the Vera index, so .gitignore and .veraignore rules apply.\n\n\
                      Source files are scanned first by default. Use `--scope docs` or \
                      `--scope runtime` when you are targeting prose or extracted runtime \
                      trees. Add `--include-generated` to scan minified/generated files.\n\n\
                      Examples:\n  \
                      vera grep \"fn\\s+main\"              # Find main functions\n  \
                      vera grep \"TODO|FIXME\" -i           # Case-insensitive\n  \
                      vera grep \"keybind\" --scope docs    # Search docs first\n  \
                      vera grep \"impl.*Display\" -n 5      # 5 results\n  \
                      vera grep \"use std::\" --context 0   # No context lines")]
    Grep {
        /// Regex pattern to search for.
        pattern: String,

        /// Maximum number of results (default: 20).
        #[arg(long, short = 'n')]
        limit: Option<usize>,

        /// Case-insensitive matching.
        #[arg(long, short = 'i')]
        ignore_case: bool,

        /// Number of context lines before and after each match (default: 2).
        #[arg(long, default_value = "2")]
        context: usize,

        /// Restrict results to a coarse corpus scope.
        #[arg(long, value_parser = ["source", "docs", "runtime", "all"])]
        scope: Option<String>,

        /// Include generated or minified files such as dist bundles.
        #[arg(long)]
        include_generated: bool,
    },

    /// Find symbols with no callers (potential dead code).
    ///
    /// Scans the call graph for functions/methods that are never called.
    /// Excludes common entry points (main, new, default, etc.).
    ///
    /// Examples:
    ///   vera dead-code
    ///   vera dead-code --json
    DeadCode,

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

    /// Watch a project directory and auto-update the index on file changes.
    ///
    /// Starts a background file watcher that triggers incremental index updates
    /// when source files change. Useful for long coding sessions where you want
    /// the index to stay fresh without manual `vera update` calls.
    ///
    /// Requires an existing index (run `vera index <path>` first).
    /// Blocks until interrupted with Ctrl-C.
    ///
    /// Examples:
    ///   vera watch .
    ///   vera watch /path/to/repo
    #[command(
        long_about = "Watch a project directory and auto-update the index on file changes.\n\n\
                      Starts a background file watcher that triggers incremental index updates \n\
                      when source files change. Changes are debounced (2s) to avoid redundant \n\
                      updates during rapid edits.\n\n\
                      Requires an existing index (run `vera index <path>` first). \n\
                      Blocks until interrupted with Ctrl-C.\n\n\
                      Examples:\n  \
                      vera watch .                   # Watch current directory\n  \
                      vera watch /path/to/repo       # Watch a specific repo\n  \
                      vera watch . --json            # JSON status output"
    )]
    Watch {
        /// Path to the directory to watch.
        path: String,
    },

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

    vera_core::init_tls();
    let cli = Cli::parse();
    if let Err(err) = state::apply_saved_env() {
        eprintln!("Error: {err:#}");
        process::exit(1);
    }

    let show_nudges = !matches!(
        cli.command,
        Commands::Mcp | Commands::Agent { .. } | Commands::Uninstall | Commands::Upgrade { .. }
    ) && !cli.json;

    let result = match cli.command {
        Commands::Mcp => {
            tracing::info!("starting MCP server");
            commands::mcp::run();
            Ok(())
        }
        Commands::Agent {
            command,
            client,
            scope,
        } => {
            tracing::info!("agent command");
            commands::agent::run(command, client, scope, cli.json)
        }
        Commands::Setup {
            backend,
            embedding,
            api,
            index,
            yes,
        } => {
            tracing::info!("setup command");
            commands::setup::run(
                backend.explicit_backend(),
                api,
                index,
                cli.json,
                yes,
                embedding,
            )
        }
        Commands::Uninstall => {
            tracing::info!("uninstall command");
            commands::uninstall::run(cli.json)
        }
        Commands::Doctor { probe } => {
            tracing::info!("doctor command");
            commands::doctor::run(cli.json, probe)
        }
        Commands::Repair { backend, api } => {
            tracing::info!("repair command");
            commands::repair::run(backend.explicit_backend(), api, cli.json)
        }
        Commands::Upgrade { apply } => {
            tracing::info!("upgrade command");
            commands::upgrade::run(apply, cli.json)
        }
        Commands::Index {
            path,
            backend,
            exclude,
            no_ignore,
            no_default_excludes,
            verbose,
            low_vram,
        } => {
            tracing::info!(path = %path, "indexing");
            commands::index::run(
                &path,
                cli.json,
                backend.resolve(),
                exclude,
                no_ignore,
                no_default_excludes,
                verbose,
                low_vram,
            )
        }
        Commands::Search {
            query,
            lang,
            path,
            limit,
            r#type,
            scope,
            include_generated,
            deep,
            backend,
        } => {
            tracing::info!(query = %query, deep, "searching");
            let filters = vera_core::types::SearchFilters {
                language: lang,
                path_glob: path,
                symbol_type: r#type,
                scope: scope.and_then(|value| value.parse().ok()),
                include_generated: Some(include_generated),
            };
            commands::search::run(
                &query,
                limit,
                &filters,
                cli.json,
                cli.raw,
                cli.timing,
                deep,
                backend.resolve(),
            )
        }
        Commands::Update {
            path,
            backend,
            exclude,
            no_ignore,
            no_default_excludes,
        } => {
            tracing::info!(path = %path, "updating");
            commands::update::run(
                &path,
                cli.json,
                backend.resolve(),
                exclude,
                no_ignore,
                no_default_excludes,
            )
        }
        Commands::Overview => {
            tracing::info!("showing overview");
            commands::overview::run(cli.json)
        }
        Commands::References { symbol, callees } => {
            tracing::info!(symbol = %symbol, callees, "references query");
            commands::references::run(&symbol, callees, cli.json)
        }
        Commands::Grep {
            pattern,
            limit,
            ignore_case,
            context,
            scope,
            include_generated,
        } => {
            tracing::info!(pattern = %pattern, "grep");
            let filters = vera_core::types::SearchFilters {
                scope: scope.and_then(|value| value.parse().ok()),
                include_generated: Some(include_generated),
                ..Default::default()
            };
            commands::grep::run(
                &pattern,
                limit,
                ignore_case,
                context,
                &filters,
                cli.json,
                cli.raw,
            )
        }
        Commands::DeadCode => {
            tracing::info!("dead code analysis");
            commands::references::run_dead_code(cli.json)
        }
        Commands::Stats => {
            tracing::info!("showing stats");
            commands::stats::run(cli.json)
        }
        Commands::Config { args } => {
            tracing::info!("config command");
            commands::config::run(&args, cli.json)
        }
        Commands::Watch { path } => {
            tracing::info!(path = %path, "watching");
            commands::watch::run(&path, cli.json)
        }
    };

    // Print update hints after the command runs (skip for MCP/agent/uninstall).
    if show_nudges {
        update_check::print_nudges();
    }

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
    fn cli_parses_agent_install_command() {
        let cli = Cli::parse_from(["vera", "agent", "install", "--client", "codex"]);
        match cli.command {
            Commands::Agent {
                command,
                client,
                scope,
                ..
            } => {
                assert_eq!(command, commands::agent::AgentCommand::Install);
                assert_eq!(client, Some(commands::agent::AgentClient::Codex));
                assert_eq!(scope, None);
            }
            _ => panic!("expected Agent command"),
        }
    }

    #[test]
    fn cli_parses_setup_command() {
        let cli = Cli::parse_from(["vera", "setup", "--local", "--index", "."]);
        match cli.command {
            Commands::Setup {
                backend,
                embedding,
                api,
                index,
                ..
            } => {
                assert!(backend.local);
                assert!(!embedding.code_rank_embed);
                assert!(!api);
                assert_eq!(index, Some(".".to_string()));
            }
            _ => panic!("expected Setup command"),
        }
    }

    #[test]
    fn cli_parses_onnx_jina_cpu_flag() {
        let cli = Cli::parse_from(["vera", "index", ".", "--onnx-jina-cpu"]);
        match cli.command {
            Commands::Index { backend, .. } => {
                assert!(backend.onnx_jina_cpu);
                assert!(!backend.local);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn cli_parses_onnx_jina_cuda_flag() {
        let cli = Cli::parse_from(["vera", "search", "test", "--onnx-jina-cuda"]);
        match cli.command {
            Commands::Search { backend, .. } => {
                assert!(backend.onnx_jina_cuda);
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_local_flag_still_works() {
        // --local is a hidden backwards-compat alias for --onnx-jina-cpu
        let cli = Cli::parse_from(["vera", "index", ".", "--local"]);
        match cli.command {
            Commands::Index { backend, .. } => {
                assert!(backend.local);
                assert!(!backend.onnx_jina_cpu);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn cli_parses_doctor_command() {
        let cli = Cli::parse_from(["vera", "doctor"]);
        assert!(matches!(cli.command, Commands::Doctor { probe: false }));
    }

    #[test]
    fn cli_parses_doctor_probe_command() {
        let cli = Cli::parse_from(["vera", "doctor", "--probe"]);
        assert!(matches!(cli.command, Commands::Doctor { probe: true }));
    }

    #[test]
    fn cli_parses_repair_command() {
        let cli = Cli::parse_from(["vera", "repair", "--onnx-jina-cuda"]);
        match cli.command {
            Commands::Repair { backend, .. } => assert!(backend.onnx_jina_cuda),
            _ => panic!("expected Repair command"),
        }
    }

    #[test]
    fn cli_parses_setup_code_rank_embed_flag() {
        let cli = Cli::parse_from(["vera", "setup", "--code-rank-embed", "--onnx-jina-cuda"]);
        match cli.command {
            Commands::Setup {
                backend, embedding, ..
            } => {
                assert!(backend.onnx_jina_cuda);
                assert!(embedding.code_rank_embed);
            }
            _ => panic!("expected Setup command"),
        }
    }

    #[test]
    fn cli_parses_upgrade_command() {
        let cli = Cli::parse_from(["vera", "upgrade", "--apply"]);
        assert!(matches!(cli.command, Commands::Upgrade { apply: true }));
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
    fn cli_parses_search_scope_flags() {
        let cli = Cli::parse_from([
            "vera",
            "search",
            "mod loader",
            "--scope",
            "runtime",
            "--include-generated",
        ]);
        match cli.command {
            Commands::Search {
                scope,
                include_generated,
                ..
            } => {
                assert_eq!(scope, Some("runtime".to_string()));
                assert!(include_generated);
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
    fn cli_parses_grep_scope_flags() {
        let cli = Cli::parse_from([
            "vera",
            "grep",
            "keybind",
            "--scope",
            "docs",
            "--include-generated",
        ]);
        match cli.command {
            Commands::Grep {
                scope,
                include_generated,
                ..
            } => {
                assert_eq!(scope, Some("docs".to_string()));
                assert!(include_generated);
            }
            _ => panic!("expected Grep command"),
        }
    }

    #[test]
    fn cli_parses_watch_command() {
        let cli = Cli::parse_from(["vera", "watch", "/tmp/repo"]);
        assert!(matches!(cli.command, Commands::Watch { path } if path == "/tmp/repo"));
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
