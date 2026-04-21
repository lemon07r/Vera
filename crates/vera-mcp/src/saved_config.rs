//! Load saved CLI runtime config for MCP-mode behavior.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct StoredConfig {
    #[serde(default)]
    core_config: Option<vera_core::config::VeraConfig>,
}

/// Load the saved runtime config from Vera's home config.json.
///
/// Returns default config on any read or parse failure so MCP stays usable.
pub fn load_saved_runtime_config() -> vera_core::config::VeraConfig {
    let config_path = match vera_core::local_models::vera_home_dir() {
        Ok(dir) => dir.join("config.json"),
        Err(_) => return vera_core::config::VeraConfig::default(),
    };
    load_runtime_config_from_path(&config_path)
}

fn load_runtime_config_from_path(config_path: &Path) -> vera_core::config::VeraConfig {
    let data = match std::fs::read(config_path) {
        Ok(data) => data,
        Err(_) => return vera_core::config::VeraConfig::default(),
    };
    if data.is_empty() {
        return vera_core::config::VeraConfig::default();
    }
    let stored: StoredConfig = match serde_json::from_slice(&data) {
        Ok(stored) => stored,
        Err(_) => return vera_core::config::VeraConfig::default(),
    };
    stored.core_config.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::load_runtime_config_from_path;

    #[test]
    fn load_config_missing_file_returns_default() {
        let tmp = tempfile::tempdir().unwrap();
        let config = load_runtime_config_from_path(&tmp.path().join("config.json"));
        assert_eq!(
            config.indexing.max_chunk_lines,
            vera_core::config::VeraConfig::default()
                .indexing
                .max_chunk_lines
        );
    }

    #[test]
    fn load_config_reads_core_config() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = vera_core::config::VeraConfig::default();
        cfg.indexing.max_chunk_lines = 99;
        cfg.indexing.max_chunk_bytes = 1800;
        cfg.retrieval.default_limit = 17;
        let json = serde_json::json!({ "core_config": cfg });
        std::fs::write(tmp.path().join("config.json"), json.to_string()).unwrap();

        let config = load_runtime_config_from_path(&tmp.path().join("config.json"));
        assert_eq!(config.indexing.max_chunk_lines, 99);
        assert_eq!(config.indexing.max_chunk_bytes, 1800);
        assert_eq!(config.retrieval.default_limit, 17);
    }

    #[test]
    fn load_config_no_core_config_key_returns_default() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), r#"{"backend":"api"}"#).unwrap();

        let config = load_runtime_config_from_path(&tmp.path().join("config.json"));
        assert_eq!(
            config.indexing.max_chunk_lines,
            vera_core::config::VeraConfig::default()
                .indexing
                .max_chunk_lines
        );
    }
}
