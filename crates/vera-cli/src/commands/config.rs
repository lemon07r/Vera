//! `vera config` — Show or set configuration values.

use std::process;

/// Run the `vera config` command.
pub fn run(args: &[String], json_output: bool) -> anyhow::Result<()> {
    let config = vera_core::config::VeraConfig::default();

    match args.first().map(|s| s.as_str()) {
        None | Some("show") => {
            // Show full configuration.
            if json_output {
                let json = serde_json::to_string_pretty(&config)
                    .map_err(|e| anyhow::anyhow!("failed to serialize config: {e}"))?;
                println!("{json}");
            } else {
                print_human_config(&config);
            }
        }
        Some("get") => {
            let key = args.get(1).unwrap_or_else(|| {
                eprintln!(
                    "Error: missing key for `vera config get`.\n\
                     Hint: use `vera config get <key>`, \
                     e.g., `vera config get retrieval.default_limit`"
                );
                process::exit(1);
            });
            let value = get_config_value(&config, key);
            match value {
                Some(v) => {
                    if json_output {
                        println!("{v}");
                    } else {
                        println!("{key} = {v}");
                    }
                }
                None => {
                    eprintln!(
                        "Error: unknown configuration key: {key}\n\
                         Hint: run `vera config show` to see all available keys."
                    );
                    process::exit(1);
                }
            }
        }
        Some("set") => {
            let key = args.get(1);
            let value = args.get(2);
            match (key, value) {
                (Some(key), Some(value)) => {
                    // Validate the key exists.
                    if get_config_value(&config, key).is_none() {
                        eprintln!(
                            "Error: unknown configuration key: {key}\n\
                             Hint: run `vera config show` to see all available keys."
                        );
                        process::exit(1);
                    }
                    // Note: Vera currently uses default config. Persistent config
                    // file support is planned. For now, show the intended change.
                    if json_output {
                        let result = serde_json::json!({
                            "key": key,
                            "value": value,
                            "status": "acknowledged",
                            "note": "Vera currently uses default configuration. \
                                     Persistent config file support is planned."
                        });
                        println!("{}", serde_json::to_string_pretty(&result).unwrap());
                    } else {
                        println!("Note: Vera currently uses default configuration.");
                        println!("Persistent config file support is planned.");
                        println!();
                        println!("Intended change: {key} = {value}");
                    }
                }
                _ => {
                    eprintln!(
                        "Error: missing key or value for `vera config set`.\n\
                         Hint: use `vera config set <key> <value>`, \
                         e.g., `vera config set retrieval.default_limit 20`"
                    );
                    process::exit(1);
                }
            }
        }
        Some(unknown) => {
            eprintln!(
                "Error: unknown config subcommand: {unknown}\n\
                 Hint: valid subcommands are: show, get, set.\n\
                 Run `vera config --help` for details."
            );
            process::exit(1);
        }
    }

    Ok(())
}

/// Print human-readable configuration.
fn print_human_config(config: &vera_core::config::VeraConfig) {
    println!("Vera Configuration");
    println!();
    println!("  Indexing:");
    println!(
        "    max_chunk_lines           {}",
        config.indexing.max_chunk_lines
    );
    println!(
        "    max_file_size_bytes       {}",
        config.indexing.max_file_size_bytes
    );
    println!(
        "    default_excludes          {:?}",
        config.indexing.default_excludes
    );
    println!();
    println!("  Retrieval:");
    println!(
        "    default_limit             {}",
        config.retrieval.default_limit
    );
    println!("    rrf_k                     {}", config.retrieval.rrf_k);
    println!(
        "    rerank_candidates         {}",
        config.retrieval.rerank_candidates
    );
    println!(
        "    reranking_enabled         {}",
        config.retrieval.reranking_enabled
    );
    println!();
    println!("  Embedding:");
    println!(
        "    batch_size                {}",
        config.embedding.batch_size
    );
    println!(
        "    max_concurrent_requests   {}",
        config.embedding.max_concurrent_requests
    );
    println!(
        "    timeout_secs              {}",
        config.embedding.timeout_secs
    );
    println!(
        "    max_retries               {}",
        config.embedding.max_retries
    );
    println!(
        "    max_stored_dim            {}",
        config.embedding.max_stored_dim
    );
}

/// Get a configuration value by dot-notation key.
pub fn get_config_value(
    config: &vera_core::config::VeraConfig,
    key: &str,
) -> Option<serde_json::Value> {
    match key {
        "indexing.max_chunk_lines" => Some(serde_json::Value::Number(
            config.indexing.max_chunk_lines.into(),
        )),
        "indexing.max_file_size_bytes" => Some(serde_json::Value::Number(
            config.indexing.max_file_size_bytes.into(),
        )),
        "indexing.default_excludes" => serde_json::to_value(&config.indexing.default_excludes).ok(),
        "retrieval.default_limit" => Some(serde_json::Value::Number(
            config.retrieval.default_limit.into(),
        )),
        "retrieval.rrf_k" => serde_json::to_value(config.retrieval.rrf_k).ok(),
        "retrieval.rerank_candidates" => Some(serde_json::Value::Number(
            config.retrieval.rerank_candidates.into(),
        )),
        "retrieval.reranking_enabled" => {
            Some(serde_json::Value::Bool(config.retrieval.reranking_enabled))
        }
        "embedding.batch_size" => Some(serde_json::Value::Number(
            config.embedding.batch_size.into(),
        )),
        "embedding.max_concurrent_requests" => Some(serde_json::Value::Number(
            config.embedding.max_concurrent_requests.into(),
        )),
        "embedding.timeout_secs" => Some(serde_json::Value::Number(
            config.embedding.timeout_secs.into(),
        )),
        "embedding.max_retries" => Some(serde_json::Value::Number(
            config.embedding.max_retries.into(),
        )),
        "embedding.max_stored_dim" => Some(serde_json::Value::Number(
            config.embedding.max_stored_dim.into(),
        )),
        _ => None,
    }
}
