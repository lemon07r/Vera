use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use std::path::PathBuf;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;

const HUB_URL: &str = "https://huggingface.co";

/// Download a file from HuggingFace Hub using atomic writes.
pub async fn ensure_model_file(repo_id: &str, file_path: &str) -> Result<PathBuf> {
    ensure_model_file_impl(repo_id, file_path, HUB_URL, None).await
}

async fn ensure_model_file_impl(
    repo_id: &str,
    file_path: &str,
    base_url: &str,
    home_override: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let home_dir = match home_override {
        Some(p) => p.to_path_buf(),
        None => dirs::home_dir().context("Could not find home directory")?,
    };
    let models_dir = home_dir.join(".vera").join("models").join(repo_id);
    let target_path = models_dir.join(file_path);

    if target_path.exists() {
        return Ok(target_path);
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let url = format!("{}/{}/resolve/main/{}", base_url, repo_id, file_path);
    eprintln!("Downloading {}...", url);

    let client = Client::new();
    let res = client.get(&url).send().await?.error_for_status()?;
    let total_size = res.content_length();

    let temp_path = target_path.with_extension("part");
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

        fs::rename(&temp_path, &target_path).await?;
        Ok(())
    }
    .await;

    if let Err(e) = download_result {
        let _ = fs::remove_file(&temp_path).await;
        return Err(e);
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
        let home = temp_dir.path().to_path_buf();

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

        let res = ensure_model_file_impl("test-repo", "test-file.bin", &base_url, Some(&home)).await;

        assert!(res.is_err(), "Download should fail due to truncated stream");

        let target_dir = home.join(".vera").join("models").join("test-repo");
        let part_file = target_dir.join("test-file.bin.part");
        assert!(
            !part_file.exists(),
            "Partial file should be cleaned up on failure"
        );
    }
}
