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

/// ONNX Runtime version to auto-download. Using 1.24.4 for CUDA 13 support.
/// The `ort` crate (rc.11) uses `load-dynamic` so any ABI-compatible ORT works.
const ORT_VERSION: &str = "1.24.4";

static ORT_INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct LocalModelAssetStatus {
    pub name: &'static str,
    pub path: PathBuf,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedLibraryDependencyStatus {
    pub inspected_files: Vec<PathBuf>,
    pub missing_details: Vec<String>,
    pub missing_libraries: Vec<String>,
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

use crate::config::OnnxExecutionProvider;

/// Detect the system CUDA major version by running `nvcc --version` or
/// checking the CUDA_PATH environment variable. Returns None if not found.
fn detect_cuda_major_version() -> Option<u32> {
    // Try nvidia-smi first (works even without toolkit installed)
    if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Look for "CUDA Version: XX.Y"
        if let Some(pos) = stdout.find("CUDA Version:") {
            let rest = &stdout[pos + 14..];
            if let Some(major) = rest.trim().split('.').next() {
                if let Ok(v) = major.parse::<u32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Platform-specific ORT archive info: (archive_ext, archive_name, primary_lib_path_inside_archive, local_lib_name).
fn ort_platform_info(
    ep: OnnxExecutionProvider,
) -> Result<(&'static str, String, String, &'static str)> {
    let gpu_suffix = match ep {
        OnnxExecutionProvider::Cpu => "",
        OnnxExecutionProvider::Cuda => {
            let cuda_major = detect_cuda_major_version().unwrap_or(12);
            if cuda_major >= 13 {
                tracing::info!("detected CUDA {cuda_major}, using CUDA 13 ORT build");
                "-gpu_cuda13"
            } else {
                tracing::info!("detected CUDA {cuda_major}, using CUDA 12 ORT build");
                "-gpu"
            }
        }
        OnnxExecutionProvider::Rocm => "-rocm",
        OnnxExecutionProvider::DirectMl => "-directml",
        OnnxExecutionProvider::CoreMl => "",
        OnnxExecutionProvider::OpenVino => "-openvino",
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if matches!(ep, OnnxExecutionProvider::DirectMl) {
            anyhow::bail!("DirectML is only supported on Windows");
        }
        if matches!(ep, OnnxExecutionProvider::OpenVino | OnnxExecutionProvider::Rocm) {
            // These EPs are installed via pip wheels, not GitHub release archives.
            // Return a dummy value; `ensure_ort_library_for_ep` handles them separately.
            let base = format!("onnxruntime-linux-x64{gpu_suffix}-{ORT_VERSION}");
            return Ok((
                "tgz",
                base.clone(),
                format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"),
                "libonnxruntime.so",
            ));
        }
        let base = format!("onnxruntime-linux-x64{gpu_suffix}-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"),
            "libonnxruntime.so",
        ))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        if !matches!(ep, OnnxExecutionProvider::Cpu) {
            anyhow::bail!("Only CPU execution provider is supported on Linux aarch64");
        }
        let base = format!("onnxruntime-linux-aarch64-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.so.{ORT_VERSION}"),
            "libonnxruntime.so",
        ))
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !matches!(
            ep,
            OnnxExecutionProvider::Cpu | OnnxExecutionProvider::CoreMl
        ) {
            anyhow::bail!("Only CPU and CoreML execution providers are supported on macOS ARM");
        }
        let base = format!("onnxruntime-osx-arm64-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.{ORT_VERSION}.dylib"),
            "libonnxruntime.dylib",
        ))
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        if !matches!(ep, OnnxExecutionProvider::Cpu) {
            anyhow::bail!("Only CPU execution provider is supported on macOS x86_64");
        }
        let base = format!("onnxruntime-osx-x86_64-{ORT_VERSION}");
        Ok((
            "tgz",
            base.clone(),
            format!("{base}/lib/libonnxruntime.{ORT_VERSION}.dylib"),
            "libonnxruntime.dylib",
        ))
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        if matches!(
            ep,
            OnnxExecutionProvider::Rocm | OnnxExecutionProvider::OpenVino
        ) {
            anyhow::bail!("ROCm and OpenVINO are only supported on Linux x86_64");
        }
        let base = format!("onnxruntime-win-x64{gpu_suffix}-{ORT_VERSION}");
        Ok((
            "zip",
            base.clone(),
            format!("{base}/lib/onnxruntime.dll"),
            "onnxruntime.dll",
        ))
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    {
        let _ = (ep, gpu_suffix);
        anyhow::bail!(
            "Unsupported platform for automatic ONNX Runtime download. \
             Install ONNX Runtime manually and set ORT_DYLIB_PATH."
        )
    }
}

/// Returns the pip package name for EPs that require pip-based installation, or None
/// for EPs that have pre-built GitHub release archives.
fn pip_package_for_ep(ep: OnnxExecutionProvider) -> Option<&'static str> {
    match ep {
        OnnxExecutionProvider::OpenVino => Some("onnxruntime-openvino"),
        OnnxExecutionProvider::Rocm => Some("onnxruntime-rocm"),
        _ => None,
    }
}

/// Try installing ORT via pip into a managed venv under `~/.vera/venv/`.
/// Returns the lib directory where .so files were copied on success.
#[cfg(target_os = "linux")]
async fn try_pip_install_ort(ep: OnnxExecutionProvider, lib_dir: &std::path::Path) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    let vera_home = vera_home_dir()?;
    let venv_dir = vera_home.join("venv");

    // Find python3
    let python = find_python3().context(
        "python3 not found. Install Python 3.11+ to enable automatic ORT installation.",
    )?;

    eprintln!("Installing {pkg} via pip (this may take a minute)...");

    // Create venv if it doesn't exist
    if !venv_dir.join("bin").join("python3").exists() {
        eprintln!("  Creating virtual environment at {}...", venv_dir.display());
        let status = tokio::process::Command::new(&python)
            .args(["-m", "venv", &venv_dir.to_string_lossy()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .status()
            .await
            .context("failed to create venv")?;
        if !status.success() {
            anyhow::bail!("failed to create virtual environment at {}", venv_dir.display());
        }
    }

    let venv_pip = venv_dir.join("bin").join("pip");
    let venv_python = venv_dir.join("bin").join("python3");

    // Upgrade pip quietly, then install the package
    let _ = tokio::process::Command::new(&venv_python)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;

    eprintln!("  Running: pip install {pkg}");
    let output = tokio::process::Command::new(&venv_pip)
        .args(["install", pkg])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("failed to run pip install")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pip install {pkg} failed:\n{stderr}");
    }

    // Find and copy .so files from the installed package
    let site_packages = find_site_packages(&venv_dir)?;
    let capi_dir = site_packages.join("onnxruntime").join("capi");
    if !capi_dir.exists() {
        anyhow::bail!(
            "pip install succeeded but onnxruntime/capi/ not found in {}",
            site_packages.display()
        );
    }

    copy_so_files_from_dir(&capi_dir, lib_dir).await?;
    Ok(())
}

/// Try downloading a wheel directly from PyPI and extracting .so files.
#[cfg(target_os = "linux")]
async fn try_wheel_download_ort(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    let pypi_name = pkg.replace('-', "_");

    eprintln!("Trying direct wheel download from PyPI...");

    crate::init_tls();
    let client = Client::new();

    // Query PyPI JSON API for the latest version's wheel URLs
    let api_url = format!("https://pypi.org/pypi/{pkg}/json");
    let resp = client
        .get(&api_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()
        .context("failed to query PyPI")?;
    let body: serde_json::Value = resp.json().await?;

    // Find a manylinux x86_64 wheel
    let urls = body["urls"]
        .as_array()
        .context("unexpected PyPI response format")?;
    let wheel_url = urls
        .iter()
        .filter_map(|entry| {
            let filename = entry["filename"].as_str()?;
            if filename.contains("linux") && filename.contains("x86_64") {
                entry["url"].as_str().map(|u| u.to_string())
            } else {
                None
            }
        })
        .next()
        .context("no compatible Linux x86_64 wheel found on PyPI")?;

    let version = body["info"]["version"]
        .as_str()
        .unwrap_or("unknown");
    eprintln!("  Downloading {pypi_name} v{version} wheel...");
    eprintln!("  {wheel_url}");

    let res = client
        .get(&wheel_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    // Wheels are zip files; extract .so files from onnxruntime/capi/
    let lib_dir_owned = lib_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_wheel_libs(&bytes, &lib_dir_owned)).await??;

    Ok(())
}

/// Extract all shared libraries from `onnxruntime/capi/` inside a wheel (zip).
#[cfg(target_os = "linux")]
fn extract_wheel_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    let mut extracted = 0usize;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let path = entry.name().to_string();
        if !path.starts_with("onnxruntime/capi/") {
            continue;
        }
        let filename = path.rsplit('/').next().unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(local_name);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
        }
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("no shared libraries found in wheel");
    }
    eprintln!("  Extracted {extracted} libraries from wheel");
    Ok(())
}

/// Find a working python3 binary.
#[cfg(target_os = "linux")]
fn find_python3() -> Option<String> {
    for name in ["python3", "python"] {
        if std::process::Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Some(name.to_string());
        }
    }
    None
}

/// Find the site-packages directory inside a venv.
#[cfg(target_os = "linux")]
fn find_site_packages(venv_dir: &std::path::Path) -> Result<PathBuf> {
    let lib_dir = venv_dir.join("lib");
    if !lib_dir.exists() {
        anyhow::bail!("venv lib directory not found");
    }
    for entry in std::fs::read_dir(&lib_dir)? {
        let entry = entry?;
        let sp = entry.path().join("site-packages");
        if sp.exists() {
            return Ok(sp);
        }
    }
    anyhow::bail!("site-packages not found in venv")
}

/// Copy all .so files from a directory to the target lib directory.
#[cfg(target_os = "linux")]
async fn copy_so_files_from_dir(
    src_dir: &std::path::Path,
    dest_dir: &std::path::Path,
) -> Result<()> {
    let mut extracted = 0usize;
    let mut entries = fs::read_dir(src_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(local_name);
        fs::copy(&path, &dest).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755)).await?;
        }
        extracted += 1;
    }
    if extracted == 0 {
        anyhow::bail!("no .so files found in {}", src_dir.display());
    }
    eprintln!("  Copied {extracted} libraries from pip package");
    Ok(())
}

/// Ensure the ONNX Runtime shared library is available locally.
///
/// Returns the path to the library. Downloads it automatically if needed.
/// Respects `ORT_DYLIB_PATH` — if set, skips auto-download.
/// GPU execution providers download a different (larger) ORT build.
///
/// For OpenVINO and ROCm (no pre-built GitHub archives), tries in order:
/// 1. `pip install` into a managed venv at `~/.vera/venv/`
/// 2. Direct wheel download from PyPI
/// 3. Bail with manual instructions
pub async fn ensure_ort_library_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    let target_path = ort_library_path_for_ep(ep)?;
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(target_path);
        }
    }

    if target_path.exists() {
        return Ok(target_path);
    }

    let lib_dir = target_path
        .parent()
        .context("failed to determine ONNX Runtime directory")?
        .to_path_buf();

    fs::create_dir_all(&lib_dir).await?;

    // OpenVINO and ROCm: pip-based install with fallback chain
    #[cfg(target_os = "linux")]
    if pip_package_for_ep(ep).is_some() {
        return ensure_ort_via_pip_chain(ep, &lib_dir, &target_path).await;
    }

    // Standard path: download from GitHub releases
    let (ext, archive_name, lib_path_in_archive, local_lib_name) = ort_platform_info(ep)?;
    let is_gpu = ep != OnnxExecutionProvider::Cpu;

    let archive_filename = if ext == "tgz" {
        format!("{archive_name}.tgz")
    } else {
        format!("{archive_name}.zip")
    };
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{archive_filename}"
    );

    eprintln!("Downloading ONNX Runtime v{ORT_VERSION} ({ep})...");
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

    let lib_dir_clone = lib_dir.clone();
    let lib_path_in_archive_clone = lib_path_in_archive.clone();

    let extract_result = tokio::task::spawn_blocking(move || -> Result<()> {
        if ext == "tgz" {
            if is_gpu {
                extract_tgz_all_libs(&bytes, &lib_dir_clone)
            } else {
                extract_tgz_single(
                    &bytes,
                    &lib_path_in_archive_clone,
                    &lib_dir_clone.join(local_lib_name),
                )
            }
        } else {
            extract_zip(
                &bytes,
                &lib_path_in_archive_clone,
                &lib_dir_clone.join(local_lib_name),
            )
        }
    })
    .await?;

    if let Err(e) = extract_result {
        return Err(e).context("Failed to extract ONNX Runtime from archive");
    }

    eprintln!(
        "ONNX Runtime v{ORT_VERSION} installed to {}",
        lib_dir.display()
    );
    Ok(target_path)
}

/// Pip-based fallback chain for OpenVINO and ROCm.
#[cfg(target_os = "linux")]
async fn ensure_ort_via_pip_chain(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
    target_path: &std::path::Path,
) -> Result<PathBuf> {
    let pkg = pip_package_for_ep(ep).unwrap();

    // Option 1: pip install into managed venv
    match try_pip_install_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!("ONNX Runtime ({ep}) installed via pip to {}", lib_dir.display());
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("pip install failed, trying direct wheel download: {e:#}");
            eprintln!("  pip install failed, trying direct wheel download...");
        }
    }

    // Option 2: download wheel directly from PyPI
    match try_wheel_download_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!("ONNX Runtime ({ep}) installed via wheel to {}", lib_dir.display());
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("wheel download failed: {e:#}");
            eprintln!("  Wheel download also failed.");
        }
    }

    // Option 3: bail with manual instructions
    anyhow::bail!(
        "Could not automatically install ONNX Runtime with {ep} support.\n\
         Install manually:\n\
         \n\
         1. pip install {pkg}\n\
         2. Locate libonnxruntime.so inside the installed package:\n\
            python3 -c \"import onnxruntime; import os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))\"\n\
         3. Set ORT_DYLIB_PATH to the full path of libonnxruntime.so\n\
         4. Run `vera setup` again"
    )
}

pub fn ort_library_path_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let vera_home = vera_home_dir()?;
    let lib_dir = if ep == OnnxExecutionProvider::Cpu {
        vera_home.join("lib")
    } else {
        vera_home.join("lib").join(ep.to_string())
    };

    Ok(lib_dir.join(platform_ort_lib_name()))
}

pub fn ensure_provider_dependencies(
    ep: OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> Result<()> {
    let Some(status) = inspect_shared_library_deps(runtime_path)? else {
        return Ok(());
    };

    if status.missing_libraries.is_empty() {
        return Ok(());
    }

    let backend_name = match ep {
        OnnxExecutionProvider::Cpu => "CPU",
        OnnxExecutionProvider::Cuda => "CUDA",
        OnnxExecutionProvider::Rocm => "ROCm",
        OnnxExecutionProvider::DirectMl => "DirectML",
        OnnxExecutionProvider::CoreMl => "CoreML",
        OnnxExecutionProvider::OpenVino => "OpenVINO",
    };

    let mut message =
        format!("{backend_name} backend selected, but required libraries are missing:\n");
    for library in &status.missing_libraries {
        message.push_str(&format!("  {library}\n"));
    }
    if let Some(hint) = dependency_hint(ep) {
        message.push_str(&hint);
        message.push('\n');
    }
    message.push_str("Run `vera doctor --probe` for details.");
    anyhow::bail!("{}", message.trim_end());
}

pub fn inspect_shared_library_deps(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    inspect_shared_library_deps_impl(runtime_path)
}

#[cfg(target_os = "linux")]
fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("ldd", &["--version"]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".so");

    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let output = std::process::Command::new("ldd")
            .arg(inspected)
            .output()
            .with_context(|| format!("failed to run `ldd` on {}", inspected.display()))?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for line in text.lines().filter(|line| line.contains("not found")) {
            let library = line.split("=>").next().unwrap_or(line).trim().to_string();
            missing_details.push(format!("{file_name}: {library}"));
            missing_libraries.push(library);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "macos")]
fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("otool", &["-L", runtime_path.to_string_lossy().as_ref()]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".dylib");
    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let dependencies = macos_dependencies(inspected)?;
        let rpaths = macos_rpaths(inspected)?;
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for dependency in dependencies {
            if macos_dependency_exists(&dependency, inspected, &rpaths) {
                continue;
            }
            missing_details.push(format!("{file_name}: {dependency}"));
            missing_libraries.push(dependency);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn inspect_shared_library_deps_impl(
    _: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    Ok(None)
}

fn collect_runtime_libraries(runtime_path: &std::path::Path, suffix: &str) -> Vec<PathBuf> {
    let mut inspected_files = vec![runtime_path.to_path_buf()];
    if let Some(dir) = runtime_path.parent() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if name.starts_with("libonnxruntime")
                    && name.contains(suffix)
                    && path != runtime_path
                {
                    inspected_files.push(path);
                }
            }
        }
    }
    inspected_files.sort();
    inspected_files.dedup();
    inspected_files
}

fn command_exists(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

#[cfg(target_os = "macos")]
fn macos_dependencies(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-L", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -L` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

#[cfg(target_os = "macos")]
fn macos_rpaths(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-l", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -l` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut rpaths = Vec::new();
    let mut in_rpath = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "cmd LC_RPATH" {
            in_rpath = true;
            continue;
        }
        if in_rpath && trimmed.starts_with("path ") {
            if let Some(path) = trimmed
                .strip_prefix("path ")
                .and_then(|rest| rest.split(" (offset").next())
            {
                rpaths.push(path.trim().to_string());
            }
            in_rpath = false;
        }
    }
    Ok(rpaths)
}

#[cfg(target_os = "macos")]
fn macos_dependency_exists(
    dependency: &str,
    inspected: &std::path::Path,
    rpaths: &[String],
) -> bool {
    if dependency.starts_with("/System/Library/") || dependency.starts_with("/usr/lib/") {
        return true;
    }

    if let Some(rest) = dependency.strip_prefix("@loader_path/") {
        return inspected
            .parent()
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@executable_path/") {
        let exe_path = std::env::current_exe().ok();
        let exe_exists = exe_path
            .as_deref()
            .and_then(|exe| exe.parent())
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
        return exe_exists
            || inspected
                .parent()
                .map(|parent| parent.join(rest).exists())
                .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@rpath/") {
        return rpaths
            .iter()
            .map(|rpath| resolve_macos_rpath(rpath, inspected))
            .any(|candidate| candidate.join(rest).exists());
    }

    if dependency.starts_with('@') {
        return false;
    }

    std::path::Path::new(dependency).exists()
}

#[cfg(target_os = "macos")]
fn resolve_macos_rpath(rpath: &str, inspected: &std::path::Path) -> PathBuf {
    if let Some(rest) = rpath.strip_prefix("@loader_path/") {
        return inspected
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .join(rest);
    }

    if let Some(rest) = rpath.strip_prefix("@executable_path/") {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                return parent.join(rest);
            }
        }
    }

    PathBuf::from(rpath)
}

fn dependency_hint(ep: OnnxExecutionProvider) -> Option<String> {
    match ep {
        OnnxExecutionProvider::Cpu => None,
        OnnxExecutionProvider::Cuda => Some(match detect_cuda_major_version() {
            Some(cuda_major) => format!(
                "Install the CUDA {cuda_major} toolkit and cuDNN 9, then ensure they're on the linker path."
            ),
            None => {
                "Install the CUDA toolkit and cuDNN 9, then ensure they're on the linker path."
                    .to_string()
            }
        }),
        OnnxExecutionProvider::Rocm => {
            Some("Install the ROCm userspace libraries, then ensure they're on the linker path.".to_string())
        }
        OnnxExecutionProvider::DirectMl => {
            Some("Install the DirectML runtime and required GPU drivers.".to_string())
        }
        OnnxExecutionProvider::CoreMl => {
            Some("Verify you are running on Apple Silicon with a supported macOS version.".to_string())
        }
        OnnxExecutionProvider::OpenVino => {
            Some("Install the Intel OpenVINO runtime or compute libraries, then ensure they're on the linker path.".to_string())
        }
    }
}

fn platform_ort_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Extract a single shared library from a tgz archive (CPU builds).
fn extract_tgz_single(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.to_string_lossy() == entry_path {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    // Suffix fallback (archive structure may vary)
    let decoder2 = GzDecoder::new(data);
    let mut archive2 = tar::Archive::new(decoder2);
    let suffix = entry_path.rsplit('/').next().unwrap_or(entry_path);

    for entry in archive2.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if path_str.ends_with(suffix) && path_str.contains("/lib/") {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    anyhow::bail!("Could not find {entry_path} in ORT archive")
}

/// Extract all shared libraries from a tgz archive (GPU builds need provider libs).
fn extract_tgz_all_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = 0usize;

    for entry in archive.entries()? {
        let mut entry = entry?;
        // Skip symlinks — we want the real files
        if entry.header().entry_type() != tar::EntryType::Regular {
            continue;
        }
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if !path_str.contains("/lib/") {
            continue;
        }
        let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        // Extract .so, .dylib, .dll files (skip .pc and other non-library files)
        let is_lib =
            filename.contains(".so") || filename.ends_with(".dylib") || filename.ends_with(".dll");
        if !is_lib {
            continue;
        }
        // Normalize versioned names: libonnxruntime.so.1.23.2 → libonnxruntime.so
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(local_name);
        write_lib_file(&mut entry, &dest)?;
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("No shared libraries found in ORT archive");
    }
    Ok(())
}

/// Strip .so version suffix: "libonnxruntime.so.1.23.2" → "libonnxruntime.so"
fn strip_so_version(name: &str) -> String {
    if let Some(pos) = name.find(".so.") {
        name[..pos + 3].to_string()
    } else {
        name.to_string()
    }
}

fn write_lib_file(reader: &mut impl std::io::Read, dest: &std::path::Path) -> Result<()> {
    let mut out = std::fs::File::create(dest)?;
    std::io::copy(reader, &mut out)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
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

/// CPU-only convenience wrapper (backwards compat).
pub async fn ensure_ort_library() -> Result<PathBuf> {
    ensure_ort_library_for_ep(OnnxExecutionProvider::Cpu).await
}

/// Download the default local embedding and reranker assets, plus the ORT library.
pub async fn prefetch_default_local_models_for_ep(
    ep: OnnxExecutionProvider,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    paths.push(ensure_ort_library_for_ep(ep).await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_ONNX_FILE).await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_ONNX_DATA_FILE).await?);
    paths.push(ensure_model_file(EMBEDDING_REPO, EMBEDDING_TOKENIZER_FILE).await?);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_ONNX_FILE).await?);
    paths.push(ensure_model_file(RERANKER_REPO, RERANKER_TOKENIZER_FILE).await?);
    Ok(paths)
}

/// CPU-only convenience wrapper (backwards compat).
pub async fn prefetch_default_local_models() -> Result<Vec<PathBuf>> {
    prefetch_default_local_models_for_ep(OnnxExecutionProvider::Cpu).await
}

/// Inspect the default local assets without downloading anything.
pub fn inspect_default_local_model_files() -> Result<Vec<LocalModelAssetStatus>> {
    inspect_default_local_model_files_for_ep(OnnxExecutionProvider::Cpu)
}

/// Inspect the default local assets for a specific execution provider.
pub fn inspect_default_local_model_files_for_ep(
    ep: OnnxExecutionProvider,
) -> Result<Vec<LocalModelAssetStatus>> {
    let vera_home = vera_home_dir()?;

    let assets = [
        ("onnx-runtime", ort_library_path_for_ep(ep)?),
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
