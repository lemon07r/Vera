//! CLI handlers for `vera references` and `vera dead-code`.

use anyhow::Result;

use crate::helpers::{apply_git_scope, load_runtime_config, output_results, prepare_indexed_repo};

/// Run the `vera references <symbol>` command.
pub fn run(
    symbol: &str,
    callees: bool,
    limit: Option<usize>,
    git_scope: Option<vera_core::git_scope::GitScope>,
    json: bool,
    raw: bool,
    compact: bool,
) -> Result<()> {
    let config = load_runtime_config()?;
    let result_limit = limit.unwrap_or(20);
    let (cwd, index_dir) = prepare_indexed_repo(&config.indexing)?;

    if callees {
        let mut results = vera_core::stats::find_callees(&cwd, symbol)?;
        if let Some(scope) = git_scope.as_ref() {
            let exact_paths = vera_core::git_scope::resolve_scope(&cwd, scope)?;
            results.retain(|result| exact_paths.contains(&result.file_path));
        }
        results.truncate(result_limit);
        if json {
            println!("{}", serde_json::to_string(&results)?);
        } else if results.is_empty() {
            println!("No callees found for '{symbol}'.");
        } else {
            println!(
                "Symbols called by '{symbol}' ({} results):\n",
                results.len()
            );
            for r in &results {
                println!("  {}:{} → {}", r.file_path, r.line, r.callee);
            }
        }
    } else {
        let filters = apply_git_scope(
            &cwd,
            &vera_core::types::SearchFilters {
                scope: Some(vera_core::types::SearchScope::Source),
                include_generated: Some(false),
                ..Default::default()
            },
            git_scope.as_ref(),
        )?;
        let results =
            vera_core::retrieval::search_callers(&index_dir, symbol, result_limit, &filters)?;
        if results.is_empty() && !json && !raw {
            println!("No callers found for '{symbol}'.");
        } else {
            output_results(
                &results,
                json,
                raw,
                compact,
                config.retrieval.max_output_chars,
            );
        }
    }
    Ok(())
}

/// Run the `vera dead-code` command.
pub fn run_dead_code(json: bool) -> Result<()> {
    let config = load_runtime_config()?;
    let (cwd, _) = prepare_indexed_repo(&config.indexing)?;
    let results = vera_core::stats::find_dead_symbols(&cwd)?;

    if json {
        println!("{}", serde_json::to_string(&results)?);
    } else if results.is_empty() {
        println!("No dead code found.");
    } else {
        println!("Potentially unused symbols ({} results):\n", results.len());
        for r in &results {
            let stype = r.symbol_type.as_deref().unwrap_or("symbol");
            println!("  {}:{} {} {}", r.file_path, r.line, stype, r.symbol_name);
        }
    }
    Ok(())
}
