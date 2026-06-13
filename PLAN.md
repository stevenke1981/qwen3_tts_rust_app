# Development Plan

## Phase 1 — CLI MVP

- Build qwentts.cpp with CPU/CUDA/Vulkan/Metal backend.
- Download talker + codec GGUF files.
- Validate metadata with `llama-gguf`.
- Generate WAV through qwentts.cpp CLI from Rust.

## Phase 2 — Desktop GUI

- Add `eframe/egui` UI.
- Fields: text, language, speaker, model paths, backend, output file.
- Add generation progress log.
- Add playback button using `rodio`.

## Phase 3 — FFI runtime

- Build qwentts.cpp as shared library with `QWEN_SHARED=ON`.
- Generate Rust bindings using `bindgen`.
- Replace process wrapper with direct function calls.
- Add streaming audio callback.

## Phase 4 — Packaging

- Bundle config template.
- Add model downloader helper.
- Package Windows/macOS/Linux builds.
