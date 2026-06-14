# Qwen3 TTS Rust App TODO

Generated on 2026-06-14 after indexing the repository with
`codebase-rlm-memory-mcp` as project `cbrlm+D-qwen3_tts_rust_app`.

## Knowledge Graph Snapshot

- Indexed scope: 21 files, about 100 symbols, 81 graph edges.
- Main code hubs:
  - `src/gui.rs` - desktop UI, generation workflow, device probing, FFI fallback.
  - `src/qwen_ffi.rs` - dynamic `qwen.dll` / shared library ABI wrapper.
  - `src/main.rs` - Clap CLI, command routing, setup-script generator.
  - `src/qwentts_cli.rs` - process-based `qwen-tts` runner and shared request trait.
  - `src/config.rs` - TOML config loading.
  - `src/downloader.rs` - Hugging Face model downloader and default-model prompt.
  - `src/gguf_probe.rs` - metadata inspection for talker/codec GGUF pairs.
- Current runtime shape: Rust app shell + GGUF metadata inspection + qwentts.cpp
  process runner, with optional FFI/GUI features. This is not yet a pure native
  Rust Qwen3-TTS inference engine.

## Current Verification (as of 2026-06-14)

- `cargo test --features "gui,ffi"` passes: **24 passed, 0 warnings**.
- `cargo check` passes: **0 warnings**.
- `cargo check --features "gui,ffi"` passes: **0 warnings**.
- `cargo fmt --check` passes: **no drift**.

## Priority 0 - Correctness / Contract Fixes

- [x] Make voice cloning force the process runner until FFI actually supports
  `ref_wav` and `ref_text`.
  - Implemented: `main.rs` + `gui.rs` check `has_cloning` before FFI selection.
  - Status: `synth --ref-wav ... --ref-text ... --features ffi` uses process runner.

- [x] Honor config defaults for `language`, `device`, and `output_dir`.
  - Implemented: CLI flags changed to `Option`, config defaults applied before hardcoded.
  - Status: omitting flags uses `qwen-tts.toml`; explicit flags override config.

- [x] Fix `ensure_default_models(out_dir, repo)` to use the passed repo.
  - Implemented: `download_default_models()` now takes `repo` parameter.
  - Status: `hf_repo` in config flows through to auto-download.

- [x] Make the default qwentts.cpp binary path Windows-aware.
  - Implemented: `default_qwen_tts_bin()` searches `build/Release/qwen-tts.exe` first.
  - Status: works on Windows and Unix.

## Priority 1 - Build Hygiene

- [x] Run `cargo fmt` or apply equivalent formatting to the existing drift.
  - Status: `cargo fmt --check` passes cleanly.

- [x] Reduce warnings in default `cargo check`.
  - Added `#[allow(dead_code)]` on TtsParams, AudioData, n_gpu_layers, tts_params.
  - Status: `cargo check` has 0 warnings.

- [x] Reduce warnings in `cargo check --features "gui,ffi"`.
  - Fixed `std::mem::forget` → `Box::leak`.
  - Added `#[allow(dead_code)]` on FFI APIs (version, is_available, speaker discovery).
  - Status: `cargo check --features "gui,ffi"` has 0 warnings.

## Priority 2 - Runtime Robustness

- [x] Add a real smoke test script that verifies a WAV artifact, not just build
  success.
  - File: `scripts/smoke_synth.ps1`.
  - Checks: model files, qwen-tts binary / qwen.dll, runs Chinese prompt,
    validates WAV RIFF header.

- [x] Improve GGUF validation beyond architecture substring checks.
  - Added `qwen3tts.*` metadata key checks (type, num_speakers, codebooks, etc.).
  - Plain Qwen3 LLM GGUFs without TTS metadata now flagged with a warning.

- [x] Add process-runner tests for command argument construction without
  spawning a real qwentts.cpp binary.
  - Extracted `build_command()`, added 6 tests: basics, speaker, instruct,
    voice clone, backend env, Chinese lang.

- [x] Add FFI ABI guardrails.
  - `QwenLibrary::check_abi()` calls `qt_init_default_params`, verifies
    `abi_version >= 3`, returns clear error on old qwen.dll builds.

## Priority 3 - Product / UX

- [x] Make GUI startup consume `AppConfig` defaults instead of hard-coded
  defaults where appropriate.
  - Status: language, device, talker/codec/bin paths all read from config on launch.

- [x] Surface backend availability from the actual qwentts.cpp runtime when
  possible.
  - Added `RuntimeProbeResult` + `probe_runtime()`: loads `qwen.dll` at startup,
    cross-references OS driver checks with runtime availability.
  - Device list annotated: "⚠ driver found, runtime unconfirmed" vs "✅ driver found + FFI runtime".
  - UI indicator (🟢/🟡) in device row with hover tooltip.

- [x] Complete packaging workflow for Windows release artifacts.
  - `scripts/package.ps1`: builds release, stages binary/DLLs/config/scripts,
    creates 7z archive.
  - Zip includes: `qwen-tts-app.exe`, `qwen.dll`, `ggml*.dll`, `qwen-tts.toml.example`,
    `README.md`, `smoke_synth.ps1`, `package.ps1`.

## Recommended Milestones (all completed)

- [x] M1: Contract cleanup — cloning fallback, config defaults, repo-aware
  auto-download, formatting.
- [x] M2: Warning cleanup — default and all-feature checks become clean.
- [x] M3: Runtime verification — add smoke synthesis script and artifact checks.
- [x] M4: FFI hardening — ABI checks, explicit cloning unsupported path.
- [x] M5: Windows packaging — portable release build with reproducible verify
  steps.

## Commands Used For This Review

```powershell
cargo test
cargo check
cargo check --features "gui,ffi"
cargo fmt --check
```
