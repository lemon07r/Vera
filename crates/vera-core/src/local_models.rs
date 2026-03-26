use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;

const HUB_URL: &str = "https://huggingface.co";
const EMBEDDING_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
const EMBEDDING_ONNX_FILE: &str = "onnx/model_quantized.onnx";
const EMBEDDING_ONNX_DATA_FILE: &str = "onnx/model_quantized.onnx_data";
const EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";
const RERANKER_REPO: &str = "jinaai/jina-reranker-v2-base-multilingual";
const RERANKER_ONNX_FILE: &str = "onnx/model_quantized.onnx";
const RERANKER_TOKENIZER_FILE: &str = "tokenizer.json";

/// ONNX Runtime version to auto-download. Must match the version expected by
/// the pinned `ort` crate (currently =2.0.0-rc.11 → ORT 1.23.2).
const ORT_VERSION: &str = "1.23.2";

static ORT_INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct LocalModelAssetStatus {
    pub name: &'static str,
    pub path: PathBuf,
    pub exists: bool,
}

/// Ensure the ONNX Runtime shared library is loaded and initialized.
///
/// Accepts an optional pre-resolved library path (from `ensure_ort_library`).
/// Falls back to system library search if no path is provided.
///
/// Safe to call multiple times — only the first call takes effect.
pub fn ensure_ort_runtime(lib_path: Option<&std::path::Path>) -> Result<()> {
    let result = ORT_INIT_RESULT.get_or_init(|| {
        let lib_name = match lib_path {
            Some(p) => p.display().to_string(),
            None => ort_lib_filename(),
        };
        match ort::init_from(&lib_name) {
            Ok(builder) => {
                builder.commit();
                Ok(())
            }
            Err(e) => Err(format!(
                "ONNX Runtime shared library not found.\n\
                 Run `vera setup --local` to auto-download it, or use API mode instead.\n\
                 Original error: {e}"
            )),
        }
    });

    match result {
        Ok(()) => Ok(()),
        Err(msg) => anyhow::bail!("{msg}"),
    }
}

/// Return Vera's home directory.
///
/// Uses `VERA_HOME` when set, otherwise defaults to `~/.vera`.
pub fn vera_home_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("VERA_HOME") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    Ok(dirs::home_dir()
        .context("Could not find home directory")?
        .join(".vera"))
}

/// Get the platform-specific ONNX Runtime shared library filename.
fn ort_lib_filename() -> String {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return path;
        }
    }

    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll".to_string()
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so".to_string()
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib".to_string()
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        "libonnxruntime.so".to_string()
    }
}

/// Platform-specific ORT archive info: (archive_name, lib_path_inside_archive, local_lib_name).
fn ort_platform_info() -> Result<(&'static str, String, String, &'static str)> {
    // Returns (archive_ext, archive_name, lib_path_in_archive, local_lib_name)
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let base = format!("onnxruntime-linux-x64-{ORT_VERSION}");
        Ok(("tgz", base.clone(), format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"), "libonnxruntime.so"))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let base = format!("onnxruntime-linux-aarch64-{ORT_VERSION}");
        Ok(("tgz", base.clone(), format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"), "libonnxruntime.so"))
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let base = format!("onnxruntime-osx-arm64-{ORT_VERSION}");
        Ok(("tgz", base.clone(), format!("{base}/lib/libonnxruntime.{ORT_VERSION}.dylib"), "libonnxruntime.dylib"))
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let base = format!("onnxruntime-osx-x86_64-{ORT_VERSION}");
        Ok(("tgz", base.clone(), format!("{base}/lib/libonnxruntime.{ORT_VERSION}.dylib"), "libonnxruntime.dylib"))
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let base = format!("onnxruntime-win-x64-{ORT_VERSION}");
        Ok(("zip", base.clone(), format!("{base}/lib/onnxruntime.dll"), "onnxruntime.dll"))
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    {
        anyhow::bail!(
            "Unsupported platform for automatic ONNX Runtime download. \
             Install ONNX Runtime manually and set ORT_DYLIB_PATH."
        )
    }
}

/// Ensure the ONNX Runtime shared library is available locally.
///
/// Returns the path to the library. Downloads it automatically if needed.
/// Respects `ORT_DYLIB_PATH` — if set, skips auto-download.
pub async fn ensure_ort_library() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let lib_dir = vera_home_dir()?.join("lib");
    let (ext, archive_name, lib_path_in_archive, local_lib_name) = ort_platform_info()?;
    let target_path = lib_dir.join(local_lib_name);

    if target_path.exists() {
        return Ok(target_path);
    }

    fs::create_dir_all(&lib_dir).await?;

    let archive_filename = if ext == "tgz" {
        format!("{archive_name}.tgz")
    } else {
        format!("{archive_name}.zip")
    };
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{archive_filename}"
    );

    eprintln!("Downloading ONNX Runtime v{ORT_VERSION}...");
    eprintln!("  {url}");

    crate::init_tls();
    let client = Client::new();
    let res = client
        .get(&url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    let temp_path = target_path.with_extension(format!("part.{}", std::process::id()));
    let lib_path_in_archive_clone = lib_path_in_archive.clone();
    let temp_path_clone = temp_path.clone();

    let extract_result = tokio::task::spawn_blocking(move || -> Result<()> {
        if ext == "tgz" {
            extract_tgz(&bytes, &lib_path_in_archive_clone, &temp_path_clone)
        } else {
            extract_zip(&bytes, &lib_path_in_archive_clone, &temp_path_clone)
        }
    })
    .await?;

    if let Err(e) = extract_result {
        let _ = fs::remove_file(&temp_path).await;
        return Err(e).context("Failed to extract ONNX Runtime from archive");
    }

    if let Err(e) = fs::rename(&temp_path, &target_path).await {
        if target_path.exists() {
            let _ = fs::remove_file(&temp_path).await;
        } else {
            let _ = fs::remove_file(&temp_path).await;
            return Err(e.into());
        }
    }

    eprintln!("ONNX Runtime v{ORT_VERSION} installed to {}", target_path.display());
    Ok(target_path)
}

fn extract_tgz(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.to_string_lossy() == entry_path {
            let mut out = std::fs::File::create(dest)?;
            std::io::copy(&mut entry, &mut out)?;
            // Set executable permission on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
            }
            return Ok(());
        }
    }

    // Try suffix match as fallback (archive structure may vary)
    let decoder2 = GzDecoder::new(data);
    let mut archive2 = tar::Archive::new(decoder2);
    let suffix = entry_path.rsplit('/').next().unwrap_or(entry_path);

    for entry in archive2.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if path_str.ends_with(suffix) && path_str.contains("/lib/") {
            let mut out = std::fs::File::create(dest)?;
            std::io::copy(&mut entry, &mut out)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
            }
            return Ok(());
        }
    }

    anyhow::bail!("Could not find {entry_path} in ORT archive")
}

fn extract_zip(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    // Windows .zip extraction using tar crate's zip support is not available,
    // so we use a minimal zip reader via the `zip` crate. Since we only compile
    // this path on Windows and want to avoid an extra dependency, we fall back
    // to extracting via the system `tar` command or manual parsing.
    //
    // For simplicity, use the zip crate. But since it's not added as a dep,
    // we'll use a raw approach: download the tgz variant if available, or
    // shell out to PowerShell on Windows.
    #[cfg(target_os = "windows")]
    {
        let temp_zip = dest.with_extension("zip");
        std::fs::write(&temp_zip, data)?;
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                     $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                     $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                     if ($entry) {{ \
                         $stream = $entry.Open(); \
                         $file = [System.IO.File]::Create('{}'); \
                         $stream.CopyTo($file); \
                         $file.Close(); $stream.Close(); \
                     }}; $zip.Dispose()",
                    temp_zip.display(),
                    entry_path.replace('/', "\\"),
                    dest.display()
                ),
            ])
            .output()?;
        let _ = std::fs::remove_file(&temp_zip);
        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract ORT zip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if !dest.exists() {
            // Try with forward slashes
            let temp_zip2 = dest.with_extension("zip2");
            std::fs::write(&temp_zip2, data)?;
            let output2 = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                         $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                         $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                         if ($entry) {{ \
                             $stream = $entry.Open(); \
                             $file = [System.IO.File]::Create('{}'); \
                             $stream.CopyTo($file); \
                             $file.Close(); $stream.Close(); \
                         }}; $zip.Dispose()",
                        temp_zip2.display(),
                        entry_path,
                        dest.display()
                    ),
                ])
                .output()?;
            let _ = std::fs::remove_file(&temp_zip2);
            if !output2.status.success() || !dest.exists() {
                anyhow::bail!("Could not find {entry_path} in ORT zip archive");
            }
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (data, entry_path, dest);
        anyhow::bail!("ZIP extraction not expected on this platform")
    }
}

/// Wrap an ort error with a user-friendly message suggesting alternatives.
pub fn wrap_ort_error(e: impl std::fmt::Display) -> String {
    let err_msg = e.to_string();
    if err_msg.contains("load")
        || err_msg.contains("libonnxruntime")
        || err_msg.contains("onnxruntime")
        || err_msg.contains("dylib")
        || err_msg.contains("dll")
        || err_msg.contains(".so")
    {
        format!(
            "ONNX Runtime shared library not found.\n\
             Run `vera setup --local` to auto-download it, or use API mode instead.\n\
             Original error: {err_msg}"
        )
    } else {
        format!("Failed to initialize ONNX Runtime: {err_msg}")
    }
}

/// Download a file from HuggingFace Hub using atomic writes.
pub async fn ensure_model_file(repo_id: &str, file_path: &str) -> Result<PathBuf> {
    ensure_model_file_impl(repo_id, file_path, HUB_URL, None).await
}

/// Download the default local embedding and reranker assets, plus the ORT library.
pub async fn prefetch_default_local_models() -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    paths.push(ensure_ort_library().await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_ONNX_FILE).await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_ONNX_DATA_FILE).await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_TOKENIZER_FILE).await?);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_ONNX_FILE).await?);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_TOKENIZER_FILE).await?);
    Ok(paths)
}

/// Inspect the default local assets without downloading anything.
pub fn inspect_default_local_model_files() -> Result<Vec<LocalModelAssetStatus>> {
    let vera_home = vera_home_dir()?;

    let ort_lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    let assets = [
        ("onnx-runtime", vera_home.join("lib").join(ort_lib_name)),
        (
            "embedding-onnx",
            vera_home
                .join("models")
                .join(EMBEDDING_REPO)
                .join(EMBEDDING_ONNX_FILE),
        ),
        (
            "embedding-onnx-data",
            vera_home
                .join("models")
                .join(EMBEDDING_REPO)
                .join(EMBEDDING_ONNX_DATA_FILE),
        ),
        (
            "embedding-tokenizer",
            vera_home
                .join("models")
                .join(EMBEDDING_REPO)
                .join(EMBEDDING_TOKENIZER_FILE),
        ),
        (
            "reranker-onnx",
            vera_home
                .join("models")
                .join(RERANKER_REPO)
                .join(RERANKER_ONNX_FILE),
        ),
        (
            "reranker-tokenizer",
            vera_home
                .join("models")
                .join(RERANKER_REPO)
                .join(RERANKER_TOKENIZER_FILE),
        ),
    ];

    Ok(assets
        .into_iter()
        .map(|(name, path)| LocalModelAssetStatus {
            name,
            exists: path.exists(),
            path,
        })
        .collect())
}

async fn ensure_model_file_impl(
    repo_id: &str,
    file_path: &str,
    base_url: &str,
    home_override: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let home_dir = match home_override {
        Some(p) => p.to_path_buf(),
        None => vera_home_dir()?,
    };
    let models_dir = home_dir.join("models").join(repo_id);
    let target_path = models_dir.join(file_path);

    if target_path.exists() {
        return Ok(target_path);
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let url = format!("{}/{}/resolve/main/{}", base_url, repo_id, file_path);
    eprintln!("Downloading {}...", url);

    crate::init_tls();
    let client = Client::new();
    let res = client.get(&url).send().await?.error_for_status()?;
    let total_size = res.content_length();

    let temp_path = target_path.with_extension(format!("part.{}", std::process::id()));
    let mut file = File::create(&temp_path).await?;
    let mut stream = res.bytes_stream();
    let mut downloaded = 0;

    let download_result: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Download error: {}", e))?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(total) = total_size {
                eprint!(
                    "\rProgress: {} MB / {} MB",
                    downloaded / 1_000_000,
                    total / 1_000_000
                );
            } else {
                eprint!("\rProgress: {} MB", downloaded / 1_000_000);
            }
        }
        file.flush().await?;
        file.sync_all().await?;
        eprintln!("\nDownload complete: {}", file_path);

        if let Err(e) = fs::rename(&temp_path, &target_path).await {
            if target_path.exists() {
                // Another process won the race
                let _ = fs::remove_file(&temp_path).await;
            } else {
                return Err(e.into());
            }
        }
        Ok(())
    }
    .await;

    if let Err(e) = download_result {
        let _ = fs::remove_file(&temp_path).await;
        return Err(e).context(format!(
            "Expected path: {}. Hint: check network connection or manually place model at {}",
            target_path.display(),
            target_path.display()
        ));
    }

    Ok(target_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[tokio::test]
    async fn test_download_failure_cleanup() {
        let temp_dir = tempfile::tempdir().unwrap();
        let home = temp_dir.path().join(".vera");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Return a valid HTTP response header but truncate the body
                let response = "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nPartialData";
                let _ = stream.write_all(response.as_bytes());
                // abruptly close the connection
            }
        });

        let base_url = format!("http://127.0.0.1:{}", port);

        let res =
            ensure_model_file_impl("test-repo", "test-file.bin", &base_url, Some(&home)).await;

        assert!(res.is_err(), "Download should fail due to truncated stream");

        let target_dir = home.join("models").join("test-repo");
        let part_file = target_dir
            .join("test-file.bin")
            .with_extension(format!("part.{}", std::process::id()));
        assert!(
            !part_file.exists(),
            "Partial file should be cleaned up on failure"
        );
    }
}
