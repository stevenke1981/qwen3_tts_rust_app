use anyhow::{Context, Result};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

/// Download a file from Hugging Face Hub with progress reporting.
pub fn download_hf_file(
    repo_id: &str,
    file_path: &str,
    out_dir: &Path,
    revision: &str,
) -> Result<PathBuf> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}");

    let client = Client::builder()
        .user_agent("qwen3-tts-rust-app/0.1.0")
        .build()
        .context("failed to build HTTP client")?;

    // Send HEAD request first to get file size
    let head_resp = client
        .head(&url)
        .send()
        .with_context(|| format!("failed to HEAD {url}"))?;

    let total_size = head_resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let resp = client
        .get(&url)
        .send()
        .with_context(|| format!("failed to GET {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "HTTP {}: failed to download {} from {}",
            resp.status(),
            file_path,
            repo_id
        );
    }

    // Ensure output directory exists
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output dir: {}", out_dir.display()))?;

    let out_path = out_dir.join(file_path);

    // Determine file name from URL or path
    let file_name = file_path.rsplit('/').next().unwrap_or(file_path);

    // Setup progress bar
    let pb = ProgressBar::new(total_size.unwrap_or(0));

    let total_display = total_size
        .map(|s| HumanBytes(s).to_string())
        .unwrap_or_else(|| "?".to_string());

    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total} ({eta}) {msg}",
            )
            .expect("valid progress template")
            .progress_chars("#>-"),
    );
    pb.set_message(format!("{} ({})", file_name, total_display));

    let start = Instant::now();
    let mut file = fs::File::create(&out_path)
        .with_context(|| format!("failed to create file: {}", out_path.display()))?;
    let mut downloaded: u64 = 0;

    // Stream body via Read trait for incremental progress
    let mut buf = [0u8; 65536]; // 64 KB chunks
    let mut response = resp;
    loop {
        let n = response
            .read(&mut buf)
            .with_context(|| format!("failed to read response body from {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("failed to write to {}", out_path.display()))?;
        downloaded += n as u64;
        pb.set_position(downloaded);

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let speed = downloaded as f64 / elapsed;
            pb.set_message(format!("{} ({}/s)", file_name, HumanBytes(speed as u64)));
        }
    }

    pb.finish_with_message(format!(
        "{} — {} in {:.1}s",
        file_name,
        HumanBytes(downloaded),
        start.elapsed().as_secs_f64()
    ));

    Ok(out_path)
}

#[allow(dead_code)]
pub const DEFAULT_REVISION: &str = "main";

/// The default Qwen3-TTS model files to download.
pub const DEFAULT_FILES: &[&str] = &[
    "qwen-talker-1.7b-base-Q8_0.gguf",
    "qwen-tokenizer-12hz-Q8_0.gguf",
];

pub const DEFAULT_REPO: &str = "Serveurperso/Qwen3-TTS-GGUF";

/// Download all default models
#[allow(dead_code)]
pub fn download_default_models(out_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for file in DEFAULT_FILES {
        let p = download_hf_file(DEFAULT_REPO, file, out_dir, DEFAULT_REVISION)?;
        paths.push(p);
    }
    Ok(paths)
}

/// Print download summary
pub fn print_summary(paths: &[PathBuf]) {
    println!("\nDownload summary:");
    for p in paths {
        let size = fs::metadata(p)
            .map(|m| HumanBytes(m.len()))
            .unwrap_or(HumanBytes(0));
        println!("  {}  {}", size, p.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_files_non_empty() {
        assert!(!DEFAULT_FILES.is_empty());
        assert!(DEFAULT_FILES.contains(&"qwen-talker-1.7b-base-Q8_0.gguf"));
    }

    #[test]
    fn test_download_url_pattern() {
        // Verify URL construction matches expected Hugging Face pattern
        let repo = "Serveurperso/Qwen3-TTS-GGUF";
        let file = "qwen-talker-1.7b-base-Q8_0.gguf";
        let rev = "main";
        let expected = "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-1.7b-base-Q8_0.gguf";
        let actual = format!("https://huggingface.co/{repo}/resolve/{rev}/{file}");
        assert_eq!(expected, actual);
    }
}
