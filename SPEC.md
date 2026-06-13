# Qwen3-TTS Rust App Spec

## Goal
Build a local Rust text-to-speech app using Qwen3-TTS GGUF files.

## Runtime model
Qwen3-TTS GGUF uses a two-stage pipeline:

1. Talker GGUF: text -> 12 Hz RVQ codes.
2. Tokenizer / codec GGUF: RVQ codes -> 24 kHz mono audio.

The Rust app uses:

- `llama-gguf` for GGUF metadata inspection and future native loading.
- `qwentts.cpp` as the production runtime for Qwen3-TTS synthesis because it already implements the custom talker + MTP code predictor + codec decode path.

## MVP features

- CLI synthesis: text input to `.wav` output.
- GGUF validation / inspection.
- Device selection through `GGML_BACKEND`.
- Support modes:
  - Base: default voice or voice clone with reference WAV.
  - CustomVoice: named speakers.
  - VoiceDesign: instruction-driven voice attributes.

## Recommended model files

Default balanced quality:

- `qwen-talker-1.7b-base-Q8_0.gguf`
- `qwen-tokenizer-12hz-Q8_0.gguf`

Low VRAM:

- `qwen-talker-1.7b-base-Q4_K_M.gguf`
- `qwen-tokenizer-12hz-Q4_K_M.gguf`

## Architecture

```text
Rust CLI / future GUI
        |
        v
AppConfig + Request Builder
        |
        +-- llama-gguf GGUF Inspector
        |
        v
QwenTtsRunner
        |
        v
qwentts.cpp build/qwen-tts
        |
        v
WAV 24 kHz mono
```

## Future native path

Replace CLI wrapper with Rust FFI over qwentts.cpp public ABI:

- `qt_init_default_params`
- `qt_init`
- `qt_tts_default_params`
- `qt_synthesize`
- `qt_audio_free`
- `qt_free`

This avoids process spawning and allows streaming callbacks in the GUI.
