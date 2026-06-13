//! Desktop GUI for Qwen3-TTS using eframe/egui.
//!
//! Feature: `gui` (behind `[features]` gate).
//! ```bash
//! cargo run --features gui -- gui
//! ```

#![cfg(feature = "gui")]

use anyhow::Result;
use eframe::egui;
use rodio::{Decoder, OutputStream, Sink};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};

use crate::config::AppConfig;
use crate::qwentts_cli::{QwenTtsRequest, QwenTtsRunner, SynthesisOutput, Synthesizer};

#[cfg(feature = "ffi")]
use crate::qwen_ffi::QwenFfiRunner;

// ═══════════════════════════════════════════════════════════════
// CJK font support
// ═══════════════════════════════════════════════════════════════

/// Try to locate and load a system CJK font, then register it with egui.
/// Without this, Chinese/Japanese/Korean characters show as tofu (□).
fn setup_cjk_fonts(ctx: &egui::Context) {
    let data = lookup_system_cjk_font();
    if let Some(data) = data {
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("cjk".to_owned(), Arc::new(egui::FontData::from_owned(data)));

        // Prepend CJK font to every font family so CJK glyphs are found first
        for family in &[
            egui::FontFamily::Proportional,
            egui::FontFamily::Monospace,
        ] {
            if let Some(list) = fonts.families.get_mut(family) {
                list.insert(0, "cjk".to_owned());
            }
        }
        ctx.set_fonts(fonts);
    } else {
        // Not critical — fall back to default (latin chars only)
        eprintln!("[gui] no system CJK font found; Chinese text may not render");
    }
}

/// Platform-specific system CJK font lookup.
fn lookup_system_cjk_font() -> Option<Vec<u8>> {
    // Windows: Microsoft YaHei (微軟雅黑), always present on Chinese Windows
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\SimHei.ttf",
            r"C:\Windows\Fonts\Deng.ttf",
        ];
        for path in &candidates {
            if Path::new(path).exists() {
                return std::fs::read(path).ok();
            }
        }
    }

    // macOS: PingFang or STHeiti
    #[cfg(target_os = "macos")]
    {
        let candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        ];
        for path in &candidates {
            if Path::new(path).exists() {
                return std::fs::read(path).ok();
            }
        }
    }

    // Linux: common package paths for Noto Sans CJK / Droid Sans Fallback
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ];
        for path in &candidates {
            if Path::new(path).exists() {
                return std::fs::read(path).ok();
            }
        }
    }

    None
}

/// Thread-safe log collector
#[derive(Clone)]
struct LogCollector {
    lines: Arc<Mutex<Vec<String>>>,
}

impl LogCollector {
    fn new() -> Self {
        Self {
            lines: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn push(&self, msg: String) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(msg);
        }
    }

    fn snapshot(&self) -> Vec<String> {
        self.lines.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

#[derive(Default, PartialEq)]
enum SynthesisMode {
    #[default]
    Base,
    CustomVoice,
    VoiceDesign,
}

struct QwenTtsApp {
    // --- Input fields ---
    text: String,
    language: String,
    speaker: String,
    instruct: String,
    talker_path: String,
    codec_path: String,
    qwen_tts_bin: String,
    output_path: String,
    ref_wav_path: String,
    ref_text_path: String,
    device: String,

    // --- State ---
    mode: SynthesisMode,
    is_generating: bool,
    log: LogCollector,
    last_wav: Option<PathBuf>,

    // Audio playback
    _stream: Option<OutputStream>,
    sink: Option<Sink>,
    is_playing: bool,
}

impl Default for QwenTtsApp {
    fn default() -> Self {
        Self {
            text: String::new(),
            language: "English".into(),
            speaker: String::new(),
            instruct: String::new(),
            talker_path: "models/qwen-talker-1.7b-base-Q8_0.gguf".into(),
            codec_path: "models/qwen-tokenizer-12hz-Q8_0.gguf".into(),
            qwen_tts_bin: "qwentts.cpp/build/qwen-tts".into(),
            output_path: "output.wav".into(),
            ref_wav_path: String::new(),
            ref_text_path: String::new(),
            device: "auto".into(),
            mode: SynthesisMode::Base,
            is_generating: false,
            log: LogCollector::new(),
            last_wav: None,
            _stream: None,
            sink: None,
            is_playing: false,
        }
    }
}

impl eframe::App for QwenTtsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Qwen3-TTS Studio");
                ui.separator();

                // --- Mode selector ---
                ui.horizontal(|ui| {
                    ui.label("Mode:");
                    ui.selectable_value(&mut self.mode, SynthesisMode::Base, "Base");
                    ui.selectable_value(&mut self.mode, SynthesisMode::CustomVoice, "CustomVoice");
                    ui.selectable_value(&mut self.mode, SynthesisMode::VoiceDesign, "VoiceDesign");
                });
                ui.separator();

                // --- Text input ---
                ui.label("Text to synthesize:");
                ui.add_sized(
                    [ui.available_width(), 80.0],
                    egui::TextEdit::multiline(&mut self.text).hint_text("Enter text for TTS..."),
                );
                ui.separator();

                // --- Main fields ---
                egui::Grid::new("main_fields")
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        ui.label("Language:");
                        egui::ComboBox::from_id_salt("lang")
                            .selected_text(&self.language)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.language,
                                    "English".to_string(),
                                    "English",
                                );
                                ui.selectable_value(
                                    &mut self.language,
                                    "Chinese".to_string(),
                                    "Chinese",
                                );
                                ui.selectable_value(
                                    &mut self.language,
                                    "Japanese".to_string(),
                                    "Japanese",
                                );
                                ui.selectable_value(
                                    &mut self.language,
                                    "Korean".to_string(),
                                    "Korean",
                                );
                                ui.selectable_value(
                                    &mut self.language,
                                    "French".to_string(),
                                    "French",
                                );
                                ui.selectable_value(
                                    &mut self.language,
                                    "German".to_string(),
                                    "German",
                                );
                            });
                        ui.end_row();

                        ui.label("Device:");
                        egui::ComboBox::from_id_salt("device")
                            .selected_text(&self.device)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.device, "auto".to_string(), "Auto");
                                ui.selectable_value(&mut self.device, "CPU".to_string(), "CPU");
                                ui.selectable_value(&mut self.device, "CUDA0".to_string(), "CUDA0");
                                ui.selectable_value(
                                    &mut self.device,
                                    "Vulkan0".to_string(),
                                    "Vulkan0",
                                );
                                ui.selectable_value(&mut self.device, "Metal".to_string(), "Metal");
                            });
                        ui.end_row();

                        // Mode-specific fields
                        match self.mode {
                            SynthesisMode::CustomVoice => {
                                ui.label("Speaker:");
                                ui.text_edit_singleline(&mut self.speaker);
                                ui.end_row();
                            }
                            SynthesisMode::VoiceDesign => {
                                ui.label("Instruction:");
                                ui.text_edit_singleline(&mut self.instruct);
                                ui.end_row();
                            }
                            SynthesisMode::Base => {}
                        }
                    });

                ui.separator();

                // --- File paths ---
                egui::CollapsingHeader::new("Model & File Paths")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::Grid::new("file_paths")
                            .striped(true)
                            .min_col_width(80.0)
                            .show(ui, |ui| {
                                ui.label("Talker GGUF:");
                                ui.text_edit_singleline(&mut self.talker_path);
                                ui.end_row();
                                ui.label("Codec GGUF:");
                                ui.text_edit_singleline(&mut self.codec_path);
                                ui.end_row();
                                ui.label("qwen-tts binary:");
                                ui.text_edit_singleline(&mut self.qwen_tts_bin);
                                ui.end_row();
                                ui.label("Output WAV:");
                                ui.text_edit_singleline(&mut self.output_path);
                                ui.end_row();

                                if self.mode == SynthesisMode::Base {
                                    ui.label("Ref WAV (clone):");
                                    ui.text_edit_singleline(&mut self.ref_wav_path);
                                    ui.end_row();
                                    ui.label("Ref text:");
                                    ui.text_edit_singleline(&mut self.ref_text_path);
                                    ui.end_row();
                                }
                            });
                    });

                ui.separator();

                // --- Generate button ---
                let can_generate = !self.text.is_empty() && !self.is_generating;
                if ui
                    .add_enabled(can_generate, egui::Button::new("Generate Speech"))
                    .clicked()
                {
                    self.start_generation(ctx);
                }

                // Playback button
                if let Some(ref wav_path) = self.last_wav {
                    if wav_path.exists() {
                        if !self.is_playing {
                            if ui.button("Play").clicked() {
                                self.play_audio(wav_path.clone());
                            }
                        } else {
                            if ui.button("Stop").clicked() {
                                self.stop_audio();
                            }
                        }
                    }
                }

                ui.separator();

                // --- Log output ---
                ui.label("Generation Log:");
                let log_text: String = {
                    let lines = self.log.snapshot();
                    lines.join("\n")
                };
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add_sized(
                            [ui.available_width(), 200.0],
                            egui::TextEdit::multiline(&mut log_text.as_str())
                                .desired_width(f32::INFINITY)
                                .font(egui::TextStyle::Monospace),
                        );
                    });

                // Poll for generation completion
                if self.is_generating {
                    ctx.request_repaint();
                }
            });
        });
    }
}

impl QwenTtsApp {
    fn start_generation(&mut self, ctx: &egui::Context) {
        let req = QwenTtsRequest {
            text: self.text.clone(),
            out: PathBuf::from(&self.output_path),
            talker: PathBuf::from(&self.talker_path),
            codec: PathBuf::from(&self.codec_path),
            lang: self.language.clone(),
            speaker: if self.mode == SynthesisMode::CustomVoice {
                Some(self.speaker.clone())
            } else {
                None
            },
            instruct: if self.mode == SynthesisMode::VoiceDesign {
                Some(self.instruct.clone())
            } else {
                None
            },
            ref_wav: if !self.ref_wav_path.is_empty() {
                Some(PathBuf::from(&self.ref_wav_path))
            } else {
                None
            },
            ref_text: if !self.ref_text_path.is_empty() {
                Some(PathBuf::from(&self.ref_text_path))
            } else {
                None
            },
            ggml_backend: if self.device != "auto" {
                Some(self.device.clone())
            } else {
                None
            },
        };

        let runner: Box<dyn Synthesizer> = runner_from(
            Some(std::path::Path::new(&self.qwen_tts_bin)),
            std::path::Path::new(&self.talker_path),
            std::path::Path::new(&self.codec_path),
            PathBuf::from(&self.qwen_tts_bin),
        );

        let log = self.log.clone();
        let ctx_clone = ctx.clone();
        let _out_path = req.out.clone();

        self.is_generating = true;
        self.log.push("Starting generation...".into());

        thread::spawn(move || {
            let result = runner.synthesize(&req);
            match result {
                Ok(SynthesisOutput::FileWritten(path)) => {
                    log.push(format!("Generated: {}", path.display()));
                }
                Ok(SynthesisOutput::AudioData(samples)) => {
                    log.push(format!(
                        "Generated {} samples ({:.1}s)",
                        samples.len(),
                        samples.len() as f64 / 24000.0
                    ));
                }
                Err(e) => {
                    log.push(format!("Error: {e}"));
                }
            }
            ctx_clone.request_repaint();
        });
    }

    fn play_audio(&mut self, path: PathBuf) {
        match OutputStream::try_default() {
            Ok((stream, stream_handle)) => match std::fs::File::open(&path) {
                Ok(file) => match Decoder::new_wav(file) {
                    Ok(decoder) => {
                        let sink = Sink::try_new(&stream_handle).ok();
                        if let Some(sink) = sink {
                            sink.append(decoder);
                            self._stream = Some(stream);
                            self.sink = Some(sink);
                            self.is_playing = true;
                            self.log.push(format!("Playing: {}", path.display()));
                        } else {
                            self.log.push("Failed to create audio sink.".into());
                        }
                    }
                    Err(e) => {
                        self.log.push(format!("Failed to decode WAV: {e}"));
                    }
                },
                Err(e) => {
                    self.log.push(format!("Failed to open WAV file: {e}"));
                }
            },
            Err(e) => {
                self.log.push(format!("Failed to open audio output: {e}"));
            }
        }
    }

    fn stop_audio(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self._stream = None;
        self.is_playing = false;
        self.log.push("Playback stopped.".into());
    }
}

/// Run the GUI application
pub fn run_gui(cfg: AppConfig) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Qwen3-TTS Studio"),
        ..Default::default()
    };

    let mut app = QwenTtsApp::default();
    // Pre-fill from config if available
    if let Some(bin) = cfg.qwen_tts_bin {
        app.qwen_tts_bin = bin.to_string_lossy().to_string();
    }
    if let Some(talker) = cfg.talker {
        app.talker_path = talker.to_string_lossy().to_string();
    }
    if let Some(codec) = cfg.codec {
        app.codec_path = codec.to_string_lossy().to_string();
    }

    eframe::run_native(
        "Qwen3-TTS Studio",
        native_options,
        Box::new(|cc| {
            setup_cjk_fonts(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))
}

// ═══════════════════════════════════════════════════════════════
// Synthesizer runner factory
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "ffi")]
fn runner_from(
    lib_path: Option<&std::path::Path>,
    talker_path: &std::path::Path,
    codec_path: &std::path::Path,
    fallback_bin: PathBuf,
) -> Box<dyn Synthesizer> {
    match QwenFfiRunner::try_new(
        lib_path,
        talker_path.to_path_buf(),
        codec_path.to_path_buf(),
    ) {
        Ok(ffi) => Box::new(ffi) as Box<dyn Synthesizer>,
        Err(_) => Box::new(QwenTtsRunner {
            qwen_tts_bin: fallback_bin,
        }) as Box<dyn Synthesizer>,
    }
}

#[cfg(not(feature = "ffi"))]
fn runner_from(
    _lib_path: Option<&std::path::Path>,
    _talker_path: &std::path::Path,
    _codec_path: &std::path::Path,
    fallback_bin: PathBuf,
) -> Box<dyn Synthesizer> {
    Box::new(QwenTtsRunner {
        qwen_tts_bin: fallback_bin,
    }) as Box<dyn Synthesizer>
}
