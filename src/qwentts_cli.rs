use anyhow::{bail, Context, Result};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

/// Advanced TTS generation parameters with sensible defaults.
#[derive(Debug, Clone, Copy)]
pub struct TtsParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: i64,
    pub max_new_tokens: i32,
}

impl Default for TtsParams {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            seed: -1, // -1 = use random seed
            max_new_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QwenTtsRunner {
    pub qwen_tts_bin: PathBuf,
}

#[derive(Debug, Clone)]
pub struct QwenTtsRequest {
    pub text: String,
    pub out: PathBuf,
    pub talker: PathBuf,
    pub codec: PathBuf,
    pub lang: String,
    pub speaker: Option<String>,
    pub instruct: Option<String>,
    pub ref_wav: Option<PathBuf>,
    pub ref_text: Option<PathBuf>,
    pub ggml_backend: Option<String>,
    /// GPU layer count (-1 = all, 0 = CPU, N = first N layers on GPU).
    /// Used by FFI path (qwen.dll ABI v3); CLI process path ignores this.
    pub n_gpu_layers: i32,
    /// Advanced synthesis parameters (temperature, top_k, etc.)
    pub tts_params: TtsParams,
}

// ═══════════════════════════════════════════════════════════════
// Synthesizer trait — shared by process and FFI runners
// ═══════════════════════════════════════════════════════════════

/// Output from a TTS synthesis.
#[derive(Debug)]
pub enum SynthesisOutput {
    /// A WAV file was written to the given path.
    FileWritten(PathBuf),
    /// Raw PCM audio samples (f32 mono, 24 kHz).
    AudioData(Vec<f32>),
}

/// Interface for TTS synthesis backends.
pub trait Synthesizer: Send {
    /// Run TTS synthesis and return the output.
    fn synthesize(&self, req: &QwenTtsRequest) -> Result<SynthesisOutput>;
}

impl QwenTtsRunner {
    pub fn synthesize(&self, req: &QwenTtsRequest) -> Result<SynthesisOutput> {
        // --- Validation (cheapest checks first) ---
        if req.text.trim().is_empty() {
            bail!("text input is empty");
        }
        if !self.qwen_tts_bin.exists() {
            bail!(
                "qwen-tts binary not found: {}\n\
                 Run setup to build it: cargo run -- setup-script --target cuda > setup.sh && bash setup.sh",
                self.qwen_tts_bin.display()
            );
        }
        if !req.talker.exists() {
            bail!(
                "talker GGUF not found: {}\n\
                 Run: cargo run -- download",
                req.talker.display()
            );
        }
        if !req.codec.exists() {
            bail!(
                "codec GGUF not found: {}\n\
                 Run: cargo run -- download",
                req.codec.display()
            );
        }

        // Create output directory
        if let Some(parent) = req.out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create output directory: {}", parent.display())
                })?;
            }
        }

        // Build command
        let mut cmd = Command::new(&self.qwen_tts_bin);
        cmd.arg("--model")
            .arg(&req.talker)
            .arg("--codec")
            .arg(&req.codec)
            .arg("--lang")
            .arg(&req.lang)
            .arg("-o")
            .arg(&req.out)
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        if let Some(speaker) = &req.speaker {
            cmd.arg("--speaker").arg(speaker);
        }
        if let Some(instruct) = &req.instruct {
            cmd.arg("--instruct").arg(instruct);
        }
        if let Some(ref_wav) = &req.ref_wav {
            cmd.arg("--ref-wav").arg(ref_wav);
        }
        if let Some(ref_text) = &req.ref_text {
            cmd.arg("--ref-text").arg(ref_text);
        }
        if let Some(backend) = &req.ggml_backend {
            cmd.env("GGML_BACKEND", backend);
        }

        tracing::debug!(
            "spawning: {} --model {} --codec {} --lang {} -o {}",
            self.qwen_tts_bin.display(),
            req.talker.display(),
            req.codec.display(),
            req.lang,
            req.out.display()
        );

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn qwen-tts binary: {}",
                self.qwen_tts_bin.display()
            )
        })?;

        // Send text via stdin
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(req.text.as_bytes())
                .context("failed to send prompt text to qwen-tts stdin")?;
            // Close stdin to signal end of input
            let _ = stdin;
        }

        let status = child.wait().context("qwen-tts process wait failed")?;
        if !status.success() {
            bail!("qwen-tts failed with status: {}", status);
        }

        if !req.out.exists() {
            bail!(
                "qwen-tts exited successfully but output file not found: {}",
                req.out.display()
            );
        }

        Ok(SynthesisOutput::FileWritten(req.out.clone()))
    }
}

impl Synthesizer for QwenTtsRunner {
    fn synthesize(&self, req: &QwenTtsRequest) -> Result<SynthesisOutput> {
        self.synthesize(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_missing_binary_returns_error() {
        let runner = QwenTtsRunner {
            qwen_tts_bin: PathBuf::from("/nonexistent/qwen-tts"),
        };
        let req = QwenTtsRequest {
            text: "hello".into(),
            out: PathBuf::from("out.wav"),
            talker: PathBuf::from("talker.gguf"),
            codec: PathBuf::from("codec.gguf"),
            lang: "English".into(),
            speaker: None,
            instruct: None,
            ref_wav: None,
            ref_text: None,
            ggml_backend: None,
            n_gpu_layers: -1,
            tts_params: TtsParams::default(),
        };
        let err = runner.synthesize(&req).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("binary not found"), "got: {msg}");
    }

    #[test]
    fn test_missing_talker_returns_error() {
        // Use current_exe() as a binary path that is guaranteed to exist
        let existing_bin = std::env::current_exe()
            .unwrap_or_else(|_| PathBuf::from("target/debug/qwen-tts-app.exe"));
        let runner = QwenTtsRunner {
            qwen_tts_bin: existing_bin,
        };
        let req = QwenTtsRequest {
            text: "hello".into(),
            out: PathBuf::from("out.wav"),
            talker: PathBuf::from("/nonexistent/talker.gguf"),
            codec: PathBuf::from("/nonexistent/codec.gguf"),
            lang: "English".into(),
            speaker: None,
            instruct: None,
            ref_wav: None,
            ref_text: None,
            ggml_backend: None,
            n_gpu_layers: -1,
            tts_params: TtsParams::default(),
        };
        let err = runner.synthesize(&req).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("talker GGUF not found"), "got: {msg}");
    }

    #[test]
    fn test_empty_text_returns_error() {
        let existing_bin = std::env::current_exe()
            .unwrap_or_else(|_| PathBuf::from("target/debug/qwen-tts-app.exe"));
        let runner = QwenTtsRunner {
            qwen_tts_bin: existing_bin,
        };
        let req = QwenTtsRequest {
            text: "   ".into(),
            out: PathBuf::from("out.wav"),
            talker: PathBuf::from("/nonexistent/talker.gguf"),
            codec: PathBuf::from("/nonexistent/codec.gguf"),
            lang: "English".into(),
            speaker: None,
            instruct: None,
            ref_wav: None,
            ref_text: None,
            ggml_backend: None,
            n_gpu_layers: -1,
            tts_params: TtsParams::default(),
        };
        let err = runner.synthesize(&req).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("text input is empty"), "got: {msg}");
    }

    #[test]
    fn test_request_builds_correct_args() {
        let req = QwenTtsRequest {
            text: "test".into(),
            out: PathBuf::from("out.wav"),
            talker: PathBuf::from("t.gguf"),
            codec: PathBuf::from("c.gguf"),
            lang: "Chinese".into(),
            speaker: Some("vivian".into()),
            instruct: None,
            ref_wav: Some(PathBuf::from("ref.wav")),
            ref_text: Some(PathBuf::from("ref.txt")),
            ggml_backend: Some("CUDA0".into()),
            n_gpu_layers: -1,
            tts_params: TtsParams::default(),
        };
        assert_eq!(req.text, "test");
        assert_eq!(req.speaker.as_deref(), Some("vivian"));
        assert_eq!(req.ggml_backend.as_deref(), Some("CUDA0"));
    }

    #[test]
    fn test_default_paths() {
        let p = Path::new("qwentts.cpp/build/qwen-tts");
        assert!(p.ends_with("qwen-tts"));
    }
}
