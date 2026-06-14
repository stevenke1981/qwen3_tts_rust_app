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
use crate::qwentts_cli::{QwenTtsRequest, QwenTtsRunner, SynthesisOutput, Synthesizer, TtsParams};

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
    n_gpu_layers: i32,
    available_devices: Vec<String>,
    // Advanced TTS params
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    seed: i64,
    max_new_tokens: i32,

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
            n_gpu_layers: -1,
            available_devices: Vec::new(),
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            seed: -1,
            max_new_tokens: 2048,
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
                                for d in &self.available_devices {
                                    ui.selectable_value(&mut self.device, d.clone(), d);
                                }
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

                // --- Advanced TTS parameters ---
                egui::CollapsingHeader::new("Advanced TTS Settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::Grid::new("advanced_params")
                            .striped(true)
                            .min_col_width(120.0)
                            .show(ui, |ui| {
                                ui.label("Temperature:");
                                ui.add(egui::Slider::new(&mut self.temperature, 0.0..=2.0).step_by(0.01));
                                ui.end_row();

                                ui.label("Top-K:");
                                ui.add(egui::Slider::new(&mut self.top_k, 0..=100).step_by(1.0));
                                ui.end_row();

                                ui.label("Top-P:");
                                ui.add(egui::Slider::new(&mut self.top_p, 0.0..=1.0).step_by(0.01));
                                ui.end_row();

                                ui.label("Repetition Penalty:");
                                ui.add(egui::Slider::new(&mut self.repetition_penalty, 1.0..=2.0).step_by(0.01));
                                ui.end_row();

                                ui.label("Seed (-1 = random):");
                                ui.add(egui::Slider::new(&mut self.seed, -1..=999999).step_by(1.0));
                                ui.end_row();

                                ui.label("Max New Tokens:");
                                ui.add(egui::Slider::new(&mut self.max_new_tokens, 64..=8192).step_by(64.0));
                                ui.end_row();

                                // Per-mode preset info
                                ui.label("Mode Preset:");
                                let preset_hint = match self.mode {
                                    SynthesisMode::Base => "Balanced: temp 0.9, top-k 50, top-p 1.0".to_string(),
                                    SynthesisMode::CustomVoice => "Voice clone: temp 0.8, top-k 40, top-p 0.95".to_string(),
                                    SynthesisMode::VoiceDesign => "Design: temp 1.0, top-k 60, top-p 1.0".to_string(),
                                };
                                ui.label(preset_hint);
                                ui.end_row();
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
        // Auto-generate output path if it matches the default
        if self.output_path == "output.wav" || self.output_path.is_empty() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.output_path = format!("output/voice_{}.wav", ts);
        }

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
            n_gpu_layers: self.n_gpu_layers,
            tts_params: TtsParams {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                repetition_penalty: self.repetition_penalty,
                seed: self.seed,
                max_new_tokens: self.max_new_tokens,
            },
        };

        // Try FFI first (auto-search qwen.dll in cwd), fall back to
        // process-based runner (needs compiled qwen-tts binary).
        let log = self.log.clone();
        let runner: Box<dyn Synthesizer> = runner_from(
            None,  // let QwenLibrary::load search default paths
            std::path::Path::new(&self.talker_path),
            std::path::Path::new(&self.codec_path),
            PathBuf::from(&self.qwen_tts_bin),
            &log,
        );

        let ctx_clone = ctx.clone();
        let _out_path = req.out.clone();

        self.is_generating = true;
        log.push("Starting generation...".into());
        let start = std::time::Instant::now();

        thread::spawn(move || {
            let elapsed = start.elapsed();
            let result = runner.synthesize(&req);
            let total_secs = elapsed.as_secs_f64();
            match result {
                Ok(SynthesisOutput::FileWritten(path)) => {
                    log.push(format!("Generated: {} ({:.2}s)", path.display(), total_secs));
                }
                Ok(SynthesisOutput::AudioData(samples)) => {
                    let audio_secs = samples.len() as f64 / 24000.0;
                    let rtf = total_secs / audio_secs.max(0.001);
                    log.push(format!(
                        "Generated {} samples ({:.1}s audio) in {:.2}s (RTF: {:.2}x)",
                        samples.len(),
                        audio_secs,
                        total_secs,
                        rtf
                    ));
                }
                Err(e) => {
                    log.push(format!("Error: {e} (after {:.2}s)", total_secs));
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
    // ── Single-instance check ──
    if is_another_instance_running() {
        anyhow::bail!(
            "Another Qwen3-TTS Studio window is already open.\n\
             Only one instance is allowed."
        );
    }

    // ── Probe available devices ──
    let available = probe_available_devices();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title(concat!("Qwen3-TTS Studio v", env!("CARGO_PKG_VERSION"))),
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
    app.available_devices = available;
    app.log.push(format!(
        "Available devices: {}",
        app.available_devices.join(", ")
    ));
    // If the only devices are "auto" and "CPU", warn about missing GPU
    if app.available_devices.len() <= 2 {
        app.log.push(
            "ℹ️ Only CPU detected. For GPU acceleration, install NVIDIA CUDA or Vulkan drivers."
                .into(),
        );
    }

    eframe::run_native(
        concat!("Qwen3-TTS Studio v", env!("CARGO_PKG_VERSION")),
        native_options,
        Box::new(|cc| {
            setup_cjk_fonts(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))
}

// ═══════════════════════════════════════════════════════════════
// Single-instance lock (Windows named mutex)
// ═══════════════════════════════════════════════════════════════

/// Check if another GUI instance is already running.
/// Uses a Windows named mutex; on non-Windows, always returns false.
fn is_another_instance_running() -> bool {
    #[cfg(target_os = "windows")]
    {
        extern "system" {
            fn CreateMutexW(
                lpMutexAttributes: *const std::ffi::c_void,
                bInitialOwner: i32,
                lpName: *const u16,
            ) -> *mut std::ffi::c_void;
            fn CloseHandle(hObject: *mut std::ffi::c_void) -> i32;
            fn GetLastError() -> u32;
        }

        const ERROR_ALREADY_EXISTS: u32 = 183;
        // Use a well-known name to prevent conflicts with other apps
        let name: Vec<u16> = "Local\\Qwen3-TTS-Studio-InstanceLock\0"
            .encode_utf16()
            .collect();
        unsafe {
            let handle = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
            if handle.is_null() {
                return false; // can't check, allow launch
            }
            let already_exists = GetLastError() == ERROR_ALREADY_EXISTS;
            if !already_exists {
                // Keep the handle for the process lifetime so the mutex stays alive
                std::mem::forget(handle);
            } else {
                CloseHandle(handle);
            }
            already_exists
        }
    }
    #[cfg(not(target_os = "windows"))]
    false
}

// ═══════════════════════════════════════════════════════════════
// GPU device probing (OS-level DLL detection)
// ═══════════════════════════════════════════════════════════════

/// Probe available GGML backends by checking for GPU driver DLLs.
/// Always includes CPU as a fallback.
fn probe_available_devices() -> Vec<String> {
    let mut devices = Vec::new();
    devices.push("auto".to_string());
    devices.push("CPU".to_string());

    #[cfg(target_os = "windows")]
    {
        if has_system_dll("nvcuda.dll") {
            // NVIDIA CUDA driver present → CUDA backend should work
            // Try to detect multiple GPUs (optional, just report CUDA0)
            devices.push("CUDA0".to_string());
        }
        if has_system_dll("vulkan-1.dll") {
            devices.push("Vulkan0".to_string());
        }
    }
    #[cfg(target_os = "macos")]
    {
        devices.push("Metal".to_string());
    }

    devices
}

/// Check whether a given system DLL can be loaded (indicating the driver exists).
#[cfg(target_os = "windows")]
fn has_system_dll(name: &str) -> bool {
    use std::ffi::CString;
    extern "system" {
        fn LoadLibraryA(lpLibFileName: *const i8) -> *mut std::ffi::c_void;
        fn FreeLibrary(hLibModule: *mut std::ffi::c_void) -> i32;
    }
    let cname = CString::new(name).unwrap();
    unsafe {
        let handle = LoadLibraryA(cname.as_ptr());
        if !handle.is_null() {
            FreeLibrary(handle);
            true
        } else {
            false
        }
    }
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
    log: &LogCollector,
) -> Box<dyn Synthesizer> {
    match QwenFfiRunner::try_new(
        lib_path,
        talker_path.to_path_buf(),
        codec_path.to_path_buf(),
    ) {
        Ok(ffi) => {
            log.push("✅ Using qwen.dll (FFI path)".into());
            Box::new(ffi) as Box<dyn Synthesizer>
        }
        Err(e) => {
            log.push(format!("⚠️ FFI init failed (qwen.dll not found?), falling back to process runner. ({e})"));
            Box::new(QwenTtsRunner {
                qwen_tts_bin: fallback_bin,
            }) as Box<dyn Synthesizer>
        }
    }
}

#[cfg(not(feature = "ffi"))]
fn runner_from(
    _lib_path: Option<&std::path::Path>,
    _talker_path: &std::path::Path,
    _codec_path: &std::path::Path,
    fallback_bin: PathBuf,
    _log: &LogCollector,
) -> Box<dyn Synthesizer> {
    Box::new(QwenTtsRunner {
        qwen_tts_bin: fallback_bin,
    }) as Box<dyn Synthesizer>
}
