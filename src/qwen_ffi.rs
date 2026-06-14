//! FFI bindings to qwen.dll (qwentts.cpp shared library).
//!
//! Uses `libloading` to dynamically load the DLL at runtime.
//! Safe Rust wrappers insulate callers from unsafe C ABI details.
//!
//! ## Library build
//! ```text
//! cmake -DQWEN_SHARED=ON -B build
//! cmake --build build --config Release
//! ```
//! Produces `qwen.dll` (Windows), `libqwen.so` (Linux), `libqwen.dylib` (macOS).

use libloading::{Library, Symbol};
use std::{
    ffi::{CStr, CString},
    path::{Path, PathBuf},
};

// ═══════════════════════════════════════════════════════════════
// C ABI types — exact layout match with qwen.h
// ═══════════════════════════════════════════════════════════════

/// Status code returned by every fallible entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum QtStatus {
    Ok = 0,
    InvalidParams = -1,
    ModeInvalid = -2,
    GenerateFailed = -3,
    Oom = -4,
    Cancelled = -5,
}

/// Log severity.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum QtLogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

/// Initialisation parameters, matching `struct qt_init_params` from qwen.h.
/// ABI v3 adds `backend` and `n_gpu_layers` at the end.
#[repr(C)]
pub(crate) struct QtInitParamsRaw {
    abi_version: i32,
    talker_path: *const i8,
    codec_path: *const i8,
    use_fa: bool,
    clamp_fp16: bool,
    // ABI v3 — set to null / -1 for defaults
    backend: *const i8,
    n_gpu_layers: i32,
}

/// Output audio buffer.
#[repr(C)]
pub(crate) struct QtAudioRaw {
    samples: *mut f32,
    n_samples: i32,
    sample_rate: i32,
    channels: i32,
}

/// Synthesis parameters.
#[repr(C)]
pub(crate) struct QtTtsParamsRaw {
    abi_version: i32,
    // 4-byte padding (x64)
    text: *const i8,
    lang: *const i8,
    instruct: *const i8,
    speaker: *const i8,
    ref_audio_24k: *const f32,
    ref_n_samples: i32,
    // 4-byte padding
    ref_text: *const i8,
    seed: i64,
    max_new_tokens: i32,
    do_sample: bool,
    // 3-byte padding
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    subtalker_do_sample: bool,
    // 3-byte padding
    subtalker_temperature: f32,
    subtalker_top_k: i32,
    subtalker_top_p: f32,
    // 4-byte padding
    dump_dir: *const i8,
    cancel: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> bool>,
    cancel_user_data: *mut std::ffi::c_void,
    on_chunk: Option<unsafe extern "C" fn(*const f32, i32, *mut std::ffi::c_void) -> bool>,
    on_chunk_user_data: *mut std::ffi::c_void,
    codec_chunk_sec: f32,
    codec_left_context_sec: f32,
    ref_spk_emb: *const f32,
    ref_spk_dim: i32,
    // 4-byte padding
    ref_codes: *const i32,
    ref_t: i32,
}

/// Opaque handle (forward declared in qwen.h as struct qt_context).
#[repr(C)]
pub(crate) struct QtContext(std::ffi::c_void);

// ═══════════════════════════════════════════════════════════════
// Function pointer types
// ═══════════════════════════════════════════════════════════════

type FnQtVersion = unsafe extern "C" fn() -> *const i8;
type FnQtLastError = unsafe extern "C" fn() -> *const i8;
type FnQtInitDefaultParams = unsafe extern "C" fn(*mut QtInitParamsRaw);
type FnQtInit = unsafe extern "C" fn(*const QtInitParamsRaw) -> *mut QtContext;
type FnQtFree = unsafe extern "C" fn(*mut QtContext);
type FnQtAudioFree = unsafe extern "C" fn(*mut QtAudioRaw);
type FnQtTtsDefaultParams = unsafe extern "C" fn(*mut QtTtsParamsRaw);
type FnQtSynthesize =
    unsafe extern "C" fn(*mut QtContext, *const QtTtsParamsRaw, *mut QtAudioRaw) -> QtStatus;
type FnQtNumCodebooks = unsafe extern "C" fn(*const QtContext) -> i32;
type FnQtDurationSecToTokens = unsafe extern "C" fn(*const QtContext, f32) -> i32;
type FnQtNSpeakers = unsafe extern "C" fn(*const QtContext) -> i32;
type FnQtSpeakerName = unsafe extern "C" fn(*const QtContext, i32) -> *const i8;
#[allow(dead_code)]
type FnQtLogSet = unsafe extern "C" fn(
    Option<unsafe extern "C" fn(QtLogLevel, *const i8, *mut std::ffi::c_void)>,
    *mut std::ffi::c_void,
);

// ═══════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════

/// Errors that can occur when using the qwen shared library.
#[derive(Debug)]
pub enum QwenFfiError {
    /// DLL/shared library could not be loaded.
    LibraryNotFound(String),
    /// A required symbol was missing from the library (version mismatch).
    MissingSymbol(String),
    /// The library returned an error status.
    SynthesisFailed(QtStatus, String),
    /// Invalid parameters.
    InvalidParams(String),
}

impl std::fmt::Display for QwenFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QwenFfiError::LibraryNotFound(msg) => write!(f, "qwen library not found: {msg}"),
            QwenFfiError::MissingSymbol(name) => {
                write!(f, "qwen library missing symbol: {name} (version mismatch?)")
            }
            QwenFfiError::SynthesisFailed(status, msg) => {
                write!(f, "qwen synthesis failed (code={status:?}): {msg}")
            }
            QwenFfiError::InvalidParams(msg) => write!(f, "invalid parameters: {msg}"),
        }
    }
}

impl std::error::Error for QwenFfiError {}

// ═══════════════════════════════════════════════════════════════
// QwenLibrary — safe wrapper around the loaded DLL
// ═══════════════════════════════════════════════════════════════

/// A loaded qwen shared library.
///
/// Holds function pointers loaded from the DLL. All `unsafe` C FFI calls
/// are wrapped in safe methods.
///
/// # Safety
///
/// The `_lib` member keeps the shared library loaded for the lifetime of
/// this struct. Raw function pointers are only valid while the library
/// is loaded. This type ensures the two lifetimes are tied.
pub(crate) struct QwenLibrary {
    #[allow(dead_code)]
    _lib: Library,

    // Cache loaded function pointers (copied out of Symbol<Fn>)
    #[allow(dead_code)]
    qt_version: FnQtVersion,
    qt_last_error: FnQtLastError,
    qt_init_default_params: FnQtInitDefaultParams,
    qt_init: FnQtInit,
    qt_free: FnQtFree,
    qt_audio_free: FnQtAudioFree,
    qt_tts_default_params: FnQtTtsDefaultParams,
    qt_synthesize: FnQtSynthesize,
    #[allow(dead_code)]
    qt_num_codebooks: FnQtNumCodebooks,
    #[allow(dead_code)]
    qt_duration_sec_to_tokens: FnQtDurationSecToTokens,
    #[allow(dead_code)]
    qt_n_speakers: FnQtNSpeakers,
    #[allow(dead_code)]
    qt_speaker_name: FnQtSpeakerName,
}

impl QwenLibrary {
    /// Try to load `qwen` shared library from standard search paths.
    ///
    /// Searches:
    ///   1. The explicit `library_path` (if provided)
    ///   2. `./qwen.dll` / `./libqwen.so` / `./libqwen.dylib`
    ///   3. `./build/bin/qwen.dll` etc. (qwentts.cpp cmake build output)
    pub fn load(custom_path: Option<&Path>) -> Result<Self, QwenFfiError> {
        let search_paths: Vec<PathBuf> = if let Some(path) = custom_path {
            vec![path.to_path_buf()]
        } else {
            let cwd = std::env::current_dir().unwrap_or_default();
            vec![
                cwd.join(lib_name()),
                cwd.join("build").join(lib_name()),
                cwd.join("build").join("bin").join(lib_name()),
                PathBuf::from(lib_name()),
            ]
        };

        let mut last_err = None;
        for path in &search_paths {
            // Safety: libloading's Library::new is safe — it loads the native library.
            // The safety obligation is on the symbols we load from it, which we
            // constrain with correct type signatures.
            match unsafe { Library::new(path) } {
                Ok(lib) => {
                    // Safety: we just loaded the library;
                    // from_library will validate the required symbols exist.
                    return unsafe { Self::from_library(lib) };
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(QwenFfiError::LibraryNotFound(format!(
            "tried paths: {}. Last error: {last_err:?}",
            search_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        )))
    }

    /// Wrap an already-opened `Library`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib` is a valid `qwen` shared library
    /// that exports all of the qt_* symbols.
    unsafe fn from_library(lib: Library) -> Result<Self, QwenFfiError> {
        // Helper: load a named symbol and dereference to get the function
        // pointer (Symbol<Fn> automatically dereferences to the fn ptr).
        macro_rules! load_fn {
            ($lib:expr, $name:literal, $ty:ty) => {{
                let sym: Symbol<'_, $ty> = $lib
                    .get($name.as_bytes())
                    .map_err(|_| QwenFfiError::MissingSymbol($name.to_string()))?;
                *sym // copy the function pointer (function pointers are Copy)
            }};
        }

        Ok(Self {
            qt_version: load_fn!(lib, "qt_version\0", FnQtVersion),
            qt_last_error: load_fn!(lib, "qt_last_error\0", FnQtLastError),
            qt_init_default_params: load_fn!(
                lib,
                "qt_init_default_params\0",
                FnQtInitDefaultParams
            ),
            qt_init: load_fn!(lib, "qt_init\0", FnQtInit),
            qt_free: load_fn!(lib, "qt_free\0", FnQtFree),
            qt_audio_free: load_fn!(lib, "qt_audio_free\0", FnQtAudioFree),
            qt_tts_default_params: load_fn!(lib, "qt_tts_default_params\0", FnQtTtsDefaultParams),
            qt_synthesize: load_fn!(lib, "qt_synthesize\0", FnQtSynthesize),
            qt_num_codebooks: load_fn!(lib, "qt_num_codebooks\0", FnQtNumCodebooks),
            qt_duration_sec_to_tokens: load_fn!(
                lib,
                "qt_duration_sec_to_tokens\0",
                FnQtDurationSecToTokens
            ),
            qt_n_speakers: load_fn!(lib, "qt_n_speakers\0", FnQtNSpeakers),
            qt_speaker_name: load_fn!(lib, "qt_speaker_name\0", FnQtSpeakerName),
            _lib: lib,
        })
    }

    /// Minimum required ABI version: we call `backend` and `n_gpu_layers`
    /// fields (ABI v3) in `QtInitParamsRaw`. Older builds produce undefined
    /// behaviour, so we reject them upfront.
    const REQUIRED_ABI_VERSION: i32 = 3;

    /// Verify the loaded library exports at least ABI v3.
    /// Returns an error with the detected version if too old.
    pub fn check_abi(&self) -> Result<(), QwenFfiError> {
        let mut params = QtInitParamsRaw {
            abi_version: Self::REQUIRED_ABI_VERSION,
            talker_path: std::ptr::null(),
            codec_path: std::ptr::null(),
            use_fa: false,
            clamp_fp16: false,
            backend: std::ptr::null(),
            n_gpu_layers: -1,
        };
        unsafe {
            (self.qt_init_default_params)(&mut params);
        }
        if params.abi_version < Self::REQUIRED_ABI_VERSION {
            let lib_abi = params.abi_version;
            let required = Self::REQUIRED_ABI_VERSION;
            return Err(QwenFfiError::SynthesisFailed(
                QtStatus::InvalidParams,
                format!(
                    "qwen.dll ABI v{lib_abi} is too old; need v{required}. \
                     Rebuild qwentts.cpp with the latest source."
                ),
            ));
        }
        tracing::info!(
            "qwen.dll reports ABI version {}, required >= {}",
            params.abi_version,
            Self::REQUIRED_ABI_VERSION,
        );
        Ok(())
    }

    /// Return the library version string.
    #[allow(dead_code)]
    pub fn version(&self) -> &str {
        unsafe {
            let ptr = (self.qt_version)();
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }

    /// Initialise a new context.
    ///
    /// `backend` selects the GGML compute backend:
    /// - `None` → auto-detect (same as "auto")
    /// - `Some("CUDA0")`, `Some("Vulkan0")`, `Some("Metal")`, `Some("CPU")`
    ///
    /// `n_gpu_layers` controls how many layers are placed on the GPU:
    /// - `-1` → all layers (default)
    /// - `0`  → CPU only
    /// - `N`  → first N layers on GPU (advisory, follow-up)
    pub fn init(
        &self,
        talker_path: &str,
        codec_path: &str,
        use_fa: bool,
        clamp_fp16: bool,
        backend: Option<&str>,
        n_gpu_layers: i32,
    ) -> Result<*mut QtContext, QwenFfiError> {
        let talker_c = CString::new(talker_path)
            .map_err(|_| QwenFfiError::InvalidParams("talker_path contains null byte".into()))?;
        let codec_c = CString::new(codec_path)
            .map_err(|_| QwenFfiError::InvalidParams("codec_path contains null byte".into()))?;
        let backend_c = backend
            .map(|s| CString::new(s))
            .transpose()
            .map_err(|_| QwenFfiError::InvalidParams("backend contains null byte".into()))?;

        let mut params = QtInitParamsRaw {
            abi_version: 3, // QT_ABI_VERSION
            talker_path: talker_c.as_ptr(),
            codec_path: codec_c.as_ptr(),
            use_fa,
            clamp_fp16,
            backend: backend_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            n_gpu_layers,
        };

        unsafe {
            (self.qt_init_default_params)(&mut params);
            // Override paths and backend (defaults set by the lib)
            params.talker_path = talker_c.as_ptr();
            params.codec_path = codec_c.as_ptr();

            let ctx = (self.qt_init)(&params);
            if ctx.is_null() {
                let err = self.last_error();
                return Err(QwenFfiError::SynthesisFailed(QtStatus::InvalidParams, err));
            }
            Ok(ctx)
        }
    }

    /// Free a context handle.
    pub fn free(&self, ctx: *mut QtContext) {
        unsafe {
            (self.qt_free)(ctx);
        }
    }

    /// Run TTS synthesis.
    pub fn synthesize(
        &self,
        ctx: *mut QtContext,
        params: &QtTtsParamsRaw,
    ) -> Result<Vec<f32>, QwenFfiError> {
        let mut audio = QtAudioRaw {
            samples: std::ptr::null_mut(),
            n_samples: 0,
            sample_rate: 0,
            channels: 0,
        };

        let status = unsafe { (self.qt_synthesize)(ctx, params, &mut audio) };
        if status != QtStatus::Ok {
            let err = self.last_error();
            // Free any partial audio
            if !audio.samples.is_null() {
                unsafe { (self.qt_audio_free)(&mut audio) };
            }
            return Err(QwenFfiError::SynthesisFailed(status, err));
        }

        // Take ownership of the samples
        let samples = if audio.n_samples > 0 && !audio.samples.is_null() {
            let len = audio.n_samples as usize;
            let result = unsafe { std::slice::from_raw_parts(audio.samples, len) }.to_vec();
            unsafe { (self.qt_audio_free)(&mut audio) };
            result
        } else {
            Vec::new()
        };

        Ok(samples)
    }

    /// Get the last error message on the calling thread.
    fn last_error(&self) -> String {
        unsafe {
            let ptr = (self.qt_last_error)();
            if ptr.is_null() {
                return String::new();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// Keep params alive until synthesis completes (CStrings must not be dropped early,
// streaming callback wrapper must stay valid for the C call duration).
#[doc(hidden)]
pub struct OwnedTtsParams {
    pub raw: QtTtsParamsRaw,
    _text: CString,
    _lang: Option<CString>,
    _instruct: Option<CString>,
    _speaker: Option<CString>,
    _streaming_wrapper: Option<Box<CbWrapper>>,
}

impl OwnedTtsParams {
    /// Build synthesis params.
    ///
    /// If `streaming` is `Some`, sets up `on_chunk` + `on_chunk_user_data`
    /// so `qt_synthesize` emits audio chunks via the provided callback.
    #[allow(clippy::too_many_arguments)]
    /// Build synthesis params.
    ///
    /// If `tts_params` is `Some`, overrides temperature, top_k, top_p,
    /// repetition_penalty, seed, and max_new_tokens with the given values.
    /// Otherwise uses the defaults from `super::qwentts_cli::TtsParams::default()`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        lib: &QwenLibrary,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        speaker: Option<&str>,
        seed: i64,
        streaming: Option<StreamingConfig>,
        tts_params: Option<&super::qwentts_cli::TtsParams>,
    ) -> Result<Self, QwenFfiError> {
        let text_c = CString::new(text)
            .map_err(|_| QwenFfiError::InvalidParams("text contains null byte".into()))?;
        let lang_c = lang
            .map(|s| CString::new(s))
            .transpose()
            .map_err(|_| QwenFfiError::InvalidParams("lang contains null byte".into()))?;
        let instruct_c = instruct
            .map(|s| CString::new(s))
            .transpose()
            .map_err(|_| QwenFfiError::InvalidParams("instruct contains null byte".into()))?;
        let speaker_c = speaker
            .map(|s| CString::new(s))
            .transpose()
            .map_err(|_| QwenFfiError::InvalidParams("speaker contains null byte".into()))?;

        // Extract chunk config before consuming `streaming` for the callback box
        let (ccs, cls) = streaming
            .as_ref()
            .map(|s| (s.chunk_duration_sec, s.left_context_sec))
            .unwrap_or((24.0, 2.0));

        // Prepare streaming callback (heap-allocate wrapper, pass pointer to C)
        let streaming_wrapper = streaming.map(|sc| Box::new(CbWrapper { cb: sc.callback }));

        let p = tts_params.copied().unwrap_or_default();
        let default_seed = if seed == -1 { p.seed } else { seed };

        let mut raw = QtTtsParamsRaw {
            abi_version: 2,
            text: text_c.as_ptr(),
            lang: lang_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            instruct: instruct_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            speaker: speaker_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            ref_audio_24k: std::ptr::null(),
            ref_n_samples: 0,
            ref_text: std::ptr::null(),
            seed: default_seed,
            max_new_tokens: p.max_new_tokens,
            do_sample: true,
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            repetition_penalty: p.repetition_penalty,
            subtalker_do_sample: true,
            subtalker_temperature: p.temperature,
            subtalker_top_k: p.top_k,
            subtalker_top_p: p.top_p,
            dump_dir: std::ptr::null(),
            cancel: None,
            cancel_user_data: std::ptr::null_mut(),
            on_chunk: streaming_wrapper
                .as_ref()
                .map(|_| audio_chunk_trampoline as unsafe extern "C" fn(_, _, _) -> bool),
            on_chunk_user_data: streaming_wrapper
                .as_ref()
                .map_or(std::ptr::null_mut(), |w| {
                    Box::as_ref(w) as *const CbWrapper as *mut std::ffi::c_void
                }),
            codec_chunk_sec: ccs,
            codec_left_context_sec: cls,
            ref_spk_emb: std::ptr::null(),
            ref_spk_dim: 0,
            ref_codes: std::ptr::null(),
            ref_t: 0,
        };

        unsafe {
            (lib.qt_tts_default_params)(&mut raw);
        }
        // Re-override fields that default_params overwrote
        raw.text = text_c.as_ptr();
        raw.lang = lang_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        raw.instruct = instruct_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        raw.speaker = speaker_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        raw.seed = default_seed;
        raw.max_new_tokens = p.max_new_tokens;
        raw.temperature = p.temperature;
        raw.top_k = p.top_k;
        raw.top_p = p.top_p;
        raw.repetition_penalty = p.repetition_penalty;
        if streaming_wrapper.is_some() {
            raw.on_chunk = Some(audio_chunk_trampoline as unsafe extern "C" fn(_, _, _) -> bool);
        }

        Ok(Self {
            raw,
            _text: text_c,
            _lang: lang_c,
            _instruct: instruct_c,
            _speaker: speaker_c,
            _streaming_wrapper: streaming_wrapper,
        })
    }
}

/// Platform-appropriate shared library filename.
fn lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "qwen.dll"
    } else if cfg!(target_os = "macos") {
        "libqwen.dylib"
    } else {
        "libqwen.so"
    }
}

// ═══════════════════════════════════════════════════════════════
// Streaming support
// ═══════════════════════════════════════════════════════════════

/// Configuration for streaming TTS synthesis via `qt_audio_chunk_cb`.
///
/// When provided, `qt_synthesize` runs in streaming mode: audio chunks
/// are emitted through `callback` as they are decoded. The callback
/// receives mono f32 PCM samples at 24 kHz. Return `true` to continue
/// or `false` to cancel (same as `CancelCb`).
pub struct StreamingConfig {
    pub callback: Box<dyn FnMut(&[f32]) -> bool + Send>,
    pub chunk_duration_sec: f32,
    pub left_context_sec: f32,
}

/// Thin wrapper heap-allocated so the C trampoline gets a stable pointer.
struct CbWrapper {
    cb: Box<dyn FnMut(&[f32]) -> bool + Send>,
}

/// `extern "C"` trampoline called by qwen library for each audio chunk.
unsafe extern "C" fn audio_chunk_trampoline(
    samples: *const f32,
    n_samples: i32,
    user_data: *mut std::ffi::c_void,
) -> bool {
    let wrapper = &mut *(user_data as *mut CbWrapper);
    let slice = std::slice::from_raw_parts(samples, n_samples as usize);
    (wrapper.cb)(slice)
}

/// FFI-based synthesizer using the qwen shared library.
pub struct QwenFfiRunner {
    lib: QwenLibrary,
    #[allow(dead_code)]
    talker_path: PathBuf,
    #[allow(dead_code)]
    codec_path: PathBuf,
}

impl QwenFfiRunner {
    /// Try to create a new FFI runner.
    pub fn try_new(
        lib_path: Option<&Path>,
        talker_path: PathBuf,
        codec_path: PathBuf,
    ) -> Result<Self, QwenFfiError> {
        let lib = QwenLibrary::load(lib_path)?;
        lib.check_abi()?;
        Ok(Self {
            lib,
            talker_path,
            codec_path,
        })
    }

    /// Check whether qwen library can be loaded (probing).
    #[allow(dead_code)]
    pub fn is_available(lib_path: Option<&Path>) -> bool {
        QwenLibrary::load(lib_path).is_ok()
    }
}

impl super::qwentts_cli::Synthesizer for QwenFfiRunner {
    fn synthesize(
        &self,
        req: &super::qwentts_cli::QwenTtsRequest,
    ) -> anyhow::Result<super::qwentts_cli::SynthesisOutput> {
        // Validate
        if req.text.trim().is_empty() {
            anyhow::bail!("text input is empty");
        }
        if !req.talker.exists() {
            anyhow::bail!("talker GGUF not found: {}", req.talker.display());
        }
        if !req.codec.exists() {
            anyhow::bail!("codec GGUF not found: {}", req.codec.display());
        }

        // Create output directory
        if let Some(parent) = req.out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let talker_str = req.talker.to_string_lossy().to_string();
        let codec_str = req.codec.to_string_lossy().to_string();

        let ctx = self
            .lib
            .init(
                &talker_str,
                &codec_str,
                true,
                false,
                req.ggml_backend.as_deref(), // GPU backend selection
                req.n_gpu_layers,            // -1 = all, 0 = CPU, N = partial
            )
            .map_err(|e| anyhow::anyhow!("qwen init failed: {e}"))?;

        let result = (|| -> anyhow::Result<super::qwentts_cli::SynthesisOutput> {
            let params = OwnedTtsParams::build(
                &self.lib,
                &req.text,
                Some(&req.lang),
                req.instruct.as_deref(),
                req.speaker.as_deref(),
                req.tts_params.seed, // -1 = random seed
                None,                // buffered mode (no streaming callback)
                Some(&req.tts_params),
            )?;

            let samples = self.lib.synthesize(ctx, &params.raw)?;

            if samples.is_empty() {
                anyhow::bail!("qwen synthesis returned no audio data");
            }

            // Write WAV file (f32 mono PCM → WAV)
            let wav_data = encode_wav_f32(&samples, 24000);
            std::fs::write(&req.out, &wav_data)
                .map_err(|e| anyhow::anyhow!("failed to write WAV: {e}"))?;

            tracing::debug!(
                "WAV written: {} ({} samples, {} sec)",
                req.out.display(),
                samples.len(),
                samples.len() as f64 / 24000.0
            );

            Ok(super::qwentts_cli::SynthesisOutput::FileWritten(
                req.out.clone(),
            ))
        })();

        // Cleanup context
        self.lib.free(ctx);

        result
    }
}

/// Encode mono f32 PCM samples (range [-1, 1]) into a WAV file.
fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    use std::io::Write;

    let channels: u16 = 1;
    let bits_per_sample: u16 = 16; // i16 for better compatibility
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
    let block_align = channels * (bits_per_sample / 8);
    let num_samples = samples.len() as u32;
    let data_size = num_samples * u32::from(block_align);
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    wav.write_all(b"RIFF").unwrap();
    wav.write_all(&(file_size.to_le_bytes())).unwrap();
    wav.write_all(b"WAVE").unwrap();

    // fmt chunk
    wav.write_all(b"fmt ").unwrap();
    wav.write_all(&(16u32.to_le_bytes())).unwrap(); // chunk size
    wav.write_all(&(1u16.to_le_bytes())).unwrap(); // PCM
    wav.write_all(&channels.to_le_bytes()).unwrap();
    wav.write_all(&sample_rate.to_le_bytes()).unwrap();
    wav.write_all(&byte_rate.to_le_bytes()).unwrap();
    wav.write_all(&block_align.to_le_bytes()).unwrap();
    wav.write_all(&bits_per_sample.to_le_bytes()).unwrap();

    // data chunk
    wav.write_all(b"data").unwrap();
    wav.write_all(&data_size.to_le_bytes()).unwrap();

    // Convert f32 samples to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * 32767.0) as i16;
        wav.write_all(&scaled.to_le_bytes()).unwrap();
    }

    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wav_f32() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_f32(&samples, 24000);
        // Minimal checks: has RIFF and WAVE markers, data chunk
        assert!(wav.starts_with(b"RIFF"));
        assert!(wav.windows(4).any(|w| w == b"WAVE"));
        assert!(wav.windows(4).any(|w| w == b"data"));
        // 5 samples * 2 bytes = 10 bytes data
        assert_eq!(wav.len(), 44 + 10);
    }

    #[test]
    fn test_ffi_runner_detection() {
        // Without a DLL, is_available should return false
        assert!(!QwenFfiRunner::is_available(Some(std::path::Path::new(
            "/nonexistent/qwen.dll"
        ))));
    }

    #[test]
    fn test_lib_name() {
        let name = lib_name();
        assert!(name.ends_with(".dll") || name.ends_with(".so") || name.ends_with(".dylib"));
    }
}
