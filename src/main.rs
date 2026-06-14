// On Windows GUI builds, suppress the console window so double-click
// doesn't flash a cmd window. CLI output still works in a terminal.
#![cfg_attr(feature = "gui", windows_subsystem = "windows")]

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

mod config;
mod downloader;
mod gguf_probe;
mod qwentts_cli;

#[cfg(feature = "gui")]
mod gui;

#[cfg(feature = "ffi")]
mod qwen_ffi;

use config::AppConfig;
use qwentts_cli::{QwenTtsRequest, QwenTtsRunner, SynthesisOutput, Synthesizer};

#[derive(Parser, Debug)]
#[command(name = "qwen-tts-app")]
#[command(about = "Rust TTS app for Qwen3-TTS GGUF — synth, inspect, download models")]
struct Cli {
    /// Optional TOML config file. CLI flags override config values.
    #[arg(long, default_value = "qwen-tts.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate speech (text → WAV).
    Synth {
        #[arg(long)]
        text: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        talker: Option<PathBuf>,
        #[arg(long)]
        codec: Option<PathBuf>,
        #[arg(long)]
        qwen_tts_bin: Option<PathBuf>,
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        speaker: Option<String>,
        #[arg(long)]
        instruct: Option<String>,
        #[arg(long)]
        ref_wav: Option<PathBuf>,
        #[arg(long)]
        ref_text: Option<PathBuf>,
        #[arg(long, value_enum)]
        device: Option<Device>,
        #[arg(long, default_value = "-1")]
        /// GPU layers (-1=all, 0=CPU, N=first N layers on GPU). FFI path only.
        n_gpu_layers: i32,
        #[arg(long, default_value = "0.9")]
        temperature: f32,
        #[arg(long, default_value = "50")]
        top_k: i32,
        #[arg(long, default_value = "1.0")]
        top_p: f32,
        #[arg(long, default_value = "1.05")]
        repetition_penalty: f32,
        #[arg(long, default_value = "-1")]
        seed: i64,
        #[arg(long, default_value = "2048")]
        max_new_tokens: i32,
    },

    /// Inspect talker / codec GGUF metadata using llama-gguf.
    Inspect {
        #[arg(long)]
        talker: PathBuf,
        #[arg(long)]
        codec: PathBuf,
    },

    /// Download Qwen3-TTS GGUF model files from Hugging Face Hub.
    Download {
        /// Hugging Face repo ID (default: Serveurperso/Qwen3-TTS-GGUF)
        #[arg(long, default_value = downloader::DEFAULT_REPO)]
        repo: String,
        /// Specific files to download (default: talker + codec Q8_0)
        #[arg(long)]
        file: Vec<String>,
        /// Output directory (default: models/)
        #[arg(long, default_value = "models")]
        out_dir: PathBuf,
        /// Git revision / branch (default: main)
        #[arg(long, default_value = "main")]
        revision: String,
        /// List available files for the given repo (dry-run, no download)
        #[arg(long)]
        list: bool,
    },

    /// Print setup script for building qwentts.cpp and downloading models.
    SetupScript {
        #[arg(long, value_enum, default_value = "cuda")]
        target: BuildTarget,
        /// Generate Windows PowerShell script instead of bash
        #[arg(long)]
        powershell: bool,
    },

    /// Launch the desktop GUI (requires `gui` feature).
    #[cfg(feature = "gui")]
    Gui,
}

#[derive(Clone, Debug, ValueEnum)]
enum Device {
    Auto,
    Cpu,
    Cuda0,
    Vulkan0,
    Metal,
}

impl Device {
    fn backend_str(&self) -> Option<&'static str> {
        match self {
            Device::Auto => None,
            Device::Cpu => Some("CPU"),
            Device::Cuda0 => Some("CUDA0"),
            Device::Vulkan0 => Some("Vulkan0"),
            Device::Metal => Some("Metal"),
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum BuildTarget {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    All,
}

/// Returns true if the given path matches one of the built-in default model paths.
fn is_default_model_path(path: &Path) -> bool {
    let defaults: &[&str] = &[
        "models/qwen-talker-1.7b-base-Q8_0.gguf",
        "models/qwen-tokenizer-12hz-Q8_0.gguf",
    ];
    let path_str = path.to_string_lossy();
    defaults.iter().any(|d| path_str == *d)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let cfg = AppConfig::load_or_default(&cli.config)?;

    match cli.command.unwrap_or_else(|| {
        // No subcommand given → default to Gui (if compiled with gui feature)
        #[cfg(feature = "gui")]
        return Commands::Gui;
        #[cfg(not(feature = "gui"))]
        {
            eprintln!(
                "{}: no command given.\n\
                 Usage: qwen-tts-app <COMMAND>\n\
                 Try:   qwen-tts-app --help\n\
                 \n\
                 Tip: recompile with --features gui for the desktop interface.",
                env!("CARGO_PKG_NAME")
            );
            std::process::exit(1);
        }
    }) {
        Commands::Synth {
            text,
            out,
            talker,
            codec,
            qwen_tts_bin,
            lang,
            speaker,
            instruct,
            ref_wav,
            ref_text,
            device,
            n_gpu_layers,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            max_new_tokens,
        } => {
            // Apply config defaults before hardcoded fallbacks.
            // CLI-provided values (Some) always win over config.
            let out_path = out.unwrap_or_else(|| cfg.output_dir.clone());
            let lang = lang.unwrap_or_else(|| cfg.language.clone());
            let device_val = device.unwrap_or_else(|| match cfg.device.to_lowercase().as_str() {
                "cpu" => Device::Cpu,
                "cuda0" => Device::Cuda0,
                "vulkan0" => Device::Vulkan0,
                "metal" => Device::Metal,
                _ => Device::Auto,
            });
            let talker_path = talker
                .or(cfg.talker)
                .unwrap_or_else(|| PathBuf::from("models/qwen-talker-1.7b-base-Q8_0.gguf"));
            let codec_path = codec
                .or(cfg.codec)
                .unwrap_or_else(|| PathBuf::from("models/qwen-tokenizer-12hz-Q8_0.gguf"));

            // Auto-download default models if missing
            if is_default_model_path(&talker_path) || is_default_model_path(&codec_path) {
                let models_dir = talker_path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("models"));
                downloader::ensure_default_models(models_dir, &cfg.hf_repo)?;
            }

            // Voice cloning must use the process runner — FFI synthesize doesn't
            // yet support ref_wav / ref_text.
            let has_cloning = ref_wav.is_some() || ref_text.is_some();

            // FFI auto-searches qwen.dll in cwd/build/; no separate --qwen-lib flag yet.
            // The --qwen-tts-bin flag only affects the process-based fallback runner.
            let custom_lib: Option<&Path> = None;
            let fallback_bin = qwen_tts_bin
                .clone()
                .or(cfg.qwen_tts_bin.clone())
                .unwrap_or_else(|| default_qwen_tts_bin());
            let runner: Box<dyn Synthesizer> = if has_cloning {
                eprintln!("ℹ️ Voice cloning: using process runner (FFI does not support ref_wav/ref_text yet)");
                Box::new(QwenTtsRunner {
                    qwen_tts_bin: fallback_bin,
                }) as Box<dyn Synthesizer>
            } else {
                create_synth_runner(custom_lib, &talker_path, &codec_path, fallback_bin)
            };

            let req = QwenTtsRequest {
                text,
                out: out_path,
                talker: talker_path.clone(),
                codec: codec_path.clone(),
                lang,
                speaker,
                instruct,
                ref_wav,
                ref_text,
                ggml_backend: device_val.backend_str().map(str::to_string),
                n_gpu_layers,
                tts_params: qwentts_cli::TtsParams {
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    seed,
                    max_new_tokens,
                },
            };

            let output = runner.synthesize(&req)?;
            match output {
                SynthesisOutput::FileWritten(path) => {
                    println!("WAV generated: {}", path.display());
                }
                SynthesisOutput::AudioData(samples) => {
                    println!(
                        "Audio generated: {} samples ({:.1}s)",
                        samples.len(),
                        samples.len() as f64 / 24000.0
                    );
                }
            }
        }

        Commands::Inspect { talker, codec } => {
            // Auto-download default models if missing before inspection
            if is_default_model_path(&talker) || is_default_model_path(&codec) {
                let models_dir = talker
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("models"));
                downloader::ensure_default_models(models_dir, &cfg.hf_repo)?;
            }
            let info = gguf_probe::inspect_pair(&talker, &codec)?;
            println!("{}", info);
        }

        Commands::Download {
            repo,
            file,
            out_dir,
            revision,
            list,
        } => {
            if list {
                println!("Available files for {repo}:");
                for f in downloader::DEFAULT_FILES {
                    println!("  {f}");
                }
                println!("\nUse `--file <name>` to download specific files.");
                return Ok(());
            }

            let files: Vec<&str> = if file.is_empty() {
                downloader::DEFAULT_FILES.to_vec()
            } else {
                file.iter().map(|s| s.as_str()).collect()
            };

            println!("Downloading from {repo} (revision: {revision})...");
            let mut downloaded = Vec::new();
            for fname in &files {
                match downloader::download_hf_file(&repo, fname, &out_dir, &revision) {
                    Ok(path) => downloaded.push(path),
                    Err(e) => {
                        eprintln!("⚠ Failed to download {fname}: {e}");
                    }
                }
            }

            if downloaded.is_empty() {
                anyhow::bail!("No files were downloaded successfully.");
            }
            downloader::print_summary(&downloaded);
        }

        Commands::SetupScript { target, powershell } => {
            if powershell {
                print_setup_script_powershell(target);
            } else {
                print_setup_script_bash(target);
            }
        }

        #[cfg(feature = "gui")]
        Commands::Gui => {
            gui::run_gui(cfg)?;
        }
    }

    Ok(())
}

/// Generate a bash setup script (Linux / macOS / WSL).
fn print_setup_script_bash(target: BuildTarget) {
    let build = match target {
        BuildTarget::Cpu => "./buildcpu.sh",
        BuildTarget::Cuda => "./buildcuda.sh",
        BuildTarget::Vulkan => "./buildvulkan.sh",
        BuildTarget::Metal => "./buildmetal.sh",
        BuildTarget::All => "./buildall.sh",
    };

    println!(
        r#"#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen3-TTS Rust App — Setup Script (bash)
# ============================================================

echo "=== Step 1: Clone qwentts.cpp ==="
if [ ! -d "qwentts.cpp" ]; then
    git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
fi
cd qwentts.cpp
git pull --recurse-submodules

echo "=== Step 2: Build qwentts.cpp ({build_name}) ==="
{build}
cd ..

echo "=== Step 3: Download GGUF models ==="
cargo run -- download --out-dir models

echo ""
echo "=== Step 4: Verify ==="
ls -lh models/*.gguf

echo ""
echo "=== Step 5: Quick test ==="
cargo run --release -- synth \
  --text "Hello from Qwen3 TTS." \
  --out test_output.wav \
  --device auto

echo ""
echo "✅ Setup complete! Run 'cargo run -- synth --help' for more options."
"#,
        build_name = format!("{:?}", target).to_lowercase()
    );
}

/// Generate a PowerShell setup script (Windows).
fn print_setup_script_powershell(target: BuildTarget) {
    let build_script = match target {
        BuildTarget::Cpu => "./buildcpu.bat",
        BuildTarget::Cuda => "./buildcuda.bat",
        BuildTarget::Vulkan => "./buildvulkan.bat",
        BuildTarget::Metal => "./buildmetal.bat",
        BuildTarget::All => "./buildall.bat",
    };

    println!(
        r#"<#
.SYNOPSIS
    Qwen3-TTS Rust App — Setup Script (PowerShell)
.DESCRIPTION
    Builds qwentts.cpp, downloads GGUF models, and verifies the setup.
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Clone qwentts.cpp ===" -ForegroundColor Cyan
if (-not (Test-Path "qwentts.cpp")) {{
    git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
}}
Push-Location qwentts.cpp
git pull --recurse-submodules

Write-Host "=== Step 2: Build qwentts.cpp ({build_name}) ===" -ForegroundColor Cyan
# On Windows, qwentts.cpp likely uses CMake directly:
# mkdir build -Force | Out-Null
# cd build
# cmake .. -DGGML_CUDA=ON -DGGML_VULKAN=ON
# cmake --build . --config Release
# Or use the provided build script:
cd ..
if (Test-Path "qwentts.cpp/{build_script}") {{
    & "qwentts.cpp\{build_script}"
}} else {{
    Write-Host "No Windows build script found; using CMake directly..." -ForegroundColor Yellow
    Push-Location qwentts.cpp
    New-Item -ItemType Directory -Path build -Force | Out-Null
    Set-Location build
    $ggml_flags = switch ("{backend}") {{
        "cuda"   {{ "-DGGML_CUDA=ON" }}
        "vulkan" {{ "-DGGML_VULKAN=ON" }}
        "metal"  {{ "-DGGML_METAL=ON" }}
        "all"    {{ "-DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_METAL=ON" }}
        default  {{ "" }}
    }}
    cmake .. -DQWEN_SHARED=ON $ggml_flags
    cmake --build . --config Release
    Pop-Location
}}
Pop-Location

Write-Host "=== Step 3: Download GGUF models ===" -ForegroundColor Cyan
cargo run -- download --out-dir models

Write-Host "`n=== Step 4: Verify ===" -ForegroundColor Cyan
Get-ChildItem models/*.gguf | ForEach-Object {{ Write-Host ("  " + $_.Name + " (" + [math]::Round($_.Length/1MB, 1) + " MB)") }}

Write-Host "`n=== Step 5: Quick test ===" -ForegroundColor Cyan
cargo run --release -- synth `
  --text "Hello from Qwen3 TTS on Windows." `
  --out test_output.wav `
  --device auto

Write-Host "`n✅ Setup complete! Run 'cargo run -- synth --help' for more options." -ForegroundColor Green
"#,
        build_name = format!("{:?}", target).to_lowercase(),
        backend = format!("{:?}", target).to_lowercase(),
        build_script = build_script,
    );
}

#[cfg(test)]
mod main_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_is_default_model_path_matches() {
        assert!(is_default_model_path(Path::new(
            "models/qwen-talker-1.7b-base-Q8_0.gguf"
        )));
        assert!(is_default_model_path(Path::new(
            "models/qwen-tokenizer-12hz-Q8_0.gguf"
        )));
    }

    #[test]
    fn test_is_default_model_path_no_match() {
        assert!(!is_default_model_path(Path::new("models/custom-file.gguf")));
        assert!(!is_default_model_path(Path::new(
            "/absolute/path/model.gguf"
        )));
        assert!(!is_default_model_path(Path::new(
            "other-dir/qwen-talker.gguf"
        )));
    }
}

/// Default qwen-tts binary path, aware of Windows CMake build output layout.
fn default_qwen_tts_bin() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        // CMake --config Release places the binary at build/Release/qwen-tts.exe
        let candidates = [
            "qwentts.cpp/build/Release/qwen-tts.exe",
            "qwentts.cpp/build/qwen-tts.exe",
        ];
        for c in &candidates {
            if Path::new(c).exists() {
                return PathBuf::from(c);
            }
        }
        PathBuf::from(candidates[0])
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from("qwentts.cpp/build/qwen-tts")
    }
}

// ═══════════════════════════════════════════════════════════════
// Synthesizer runner factory (FFI with process fallback)
// ═══════════════════════════════════════════════════════════════

/// Create a synthesizer: try FFI shared library first, fall back to
/// process-based runner.
#[cfg(feature = "ffi")]
fn create_synth_runner(
    lib_path: Option<&Path>,
    talker_path: &Path,
    codec_path: &Path,
    fallback_bin: PathBuf,
) -> Box<dyn Synthesizer> {
    match qwen_ffi::QwenFfiRunner::try_new(
        lib_path,
        talker_path.to_path_buf(),
        codec_path.to_path_buf(),
    ) {
        Ok(ffi) => {
            tracing::info!("Using qwen shared library (FFI)");
            Box::new(ffi) as Box<dyn Synthesizer>
        }
        Err(_) => {
            tracing::info!("qwen library not found, using process-based runner");
            Box::new(QwenTtsRunner {
                qwen_tts_bin: fallback_bin,
            }) as Box<dyn Synthesizer>
        }
    }
}

#[cfg(not(feature = "ffi"))]
fn create_synth_runner(
    _lib_path: Option<&Path>,
    _talker_path: &Path,
    _codec_path: &Path,
    fallback_bin: PathBuf,
) -> Box<dyn Synthesizer> {
    Box::new(QwenTtsRunner {
        qwen_tts_bin: fallback_bin,
    }) as Box<dyn Synthesizer>
}
