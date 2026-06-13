use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub qwen_tts_bin: Option<PathBuf>,
    pub talker: Option<PathBuf>,
    pub codec: Option<PathBuf>,

    /// Default language for TTS
    #[serde(default = "default_lang")]
    pub language: String,

    /// Default device / backend
    #[serde(default = "default_device")]
    pub device: String,

    /// Output directory for generated WAV files
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,

    /// Hugging Face repo for model downloads
    #[serde(default = "default_repo")]
    pub hf_repo: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            qwen_tts_bin: None,
            talker: None,
            codec: None,
            language: default_lang(),
            device: default_device(),
            output_dir: default_output_dir(),
            hf_repo: default_repo(),
        }
    }
}

fn default_lang() -> String {
    "English".into()
}

fn default_device() -> String {
    "auto".into()
}

fn default_output_dir() -> PathBuf {
    PathBuf::from(".")
}

fn default_repo() -> String {
    "Serveurperso/Qwen3-TTS-GGUF".into()
}

impl AppConfig {
    pub fn load_or_default(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let cfg: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse TOML config: {}", path.display()))?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_nonexistent_config_returns_default() {
        let p = Path::new("/nonexistent/path/qwen-tts.toml");
        let cfg = AppConfig::load_or_default(p).unwrap();
        assert!(cfg.qwen_tts_bin.is_none());
        assert_eq!(cfg.language, "English");
        assert_eq!(cfg.device, "auto");
    }

    #[test]
    fn test_load_valid_toml() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("qwen-tts.toml");
        let toml_content = r#"
qwen_tts_bin = "my-qwen-tts.exe"
talker = "my-talker.gguf"
codec = "my-codec.gguf"
language = "Chinese"
device = "CUDA0"
"#;
        let mut f = fs::File::create(&cfg_path).unwrap();
        f.write_all(toml_content.as_bytes()).unwrap();
        let cfg = AppConfig::load_or_default(&cfg_path).unwrap();
        assert_eq!(
            cfg.qwen_tts_bin.unwrap().to_string_lossy(),
            "my-qwen-tts.exe"
        );
        assert_eq!(cfg.language, "Chinese");
        assert_eq!(cfg.device, "CUDA0");
    }

    #[test]
    fn test_invalid_toml_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("bad.toml");
        let mut f = fs::File::create(&cfg_path).unwrap();
        f.write_all(b"not valid toml {{{").unwrap();
        assert!(AppConfig::load_or_default(&cfg_path).is_err());
    }

    #[test]
    fn test_default_values() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.language, "English");
        assert_eq!(cfg.device, "auto");
        assert_eq!(cfg.output_dir, PathBuf::from("."));
    }
}
