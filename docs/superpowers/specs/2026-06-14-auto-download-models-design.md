# Auto-download Models — Design Spec

**Date:** 2026-06-14
**Status:** Approved

## Goal

When the user runs `synth` (or `inspect`) and the default model files are missing from the
`models/` directory, prompt the user for confirmation, then auto-download them from Hugging Face
Hub before proceeding with the command.

## Scope

- **Trigger commands:** `synth`, `inspect`
- **Trigger condition:** Only when the resolved model path equals the default `models/` path
- **User interaction:** Prompt with `[Y/n]` before downloading (files are 1-2 GB each)
- **Custom paths:** If user provides `--talker /custom/path.gguf`, do NOT auto-download; keep
  existing error behavior

## Current State

- `downloader.rs` has a `download_default_models()` function marked `#[allow(dead_code)]` —
  never called
- `Synth` handler in `main.rs` resolves paths: CLI arg > config > default (`models/*.gguf`)
- Runner layer (both process and FFI) checks file existence and returns an error if missing
- The `Download` subcommand already supports manual download

## Changes

### 1. `downloader.rs` — Enable dead code + add prompt function

- Remove `#[allow(dead_code)]` from `download_default_models()` and `DEFAULT_REVISION`
- Add new public function:

```rust
/// Check if default models exist; if not, prompt user and download.
/// Returns Ok(()) if models are present or were successfully downloaded.
/// Returns Err if user declines or download fails.
pub fn ensure_default_models(out_dir: &Path, repo: &str) -> Result<()>
```

This function:
1. Builds the list of expected paths (`out_dir / filename` for each `DEFAULT_FILES`)
2. If all exist, returns `Ok(())` immediately
3. If any missing, prints a summary and asks `? Download default models (2 files, ~2 GB)? [Y/n]`
4. On `Y`, calls `download_default_models()` and reports results
5. On `n`, returns `anyhow::bail!("download cancelled by user")`

### 2. `main.rs` — Wire auto-download into command handlers

In both `Commands::Synth` and `Commands::Inspect`:

```rust
// After resolving talker_path and codec_path, before creating request:
if is_default_model_path(&talker_path) || is_default_model_path(&codec_path) {
    // Check models/ parent directory
    let models_dir = talker_path.parent().unwrap_or(Path::new("models"));
    downloader::ensure_default_models(models_dir, &cfg.hf_repo)?;
}
```

Add a helper:

```rust
fn is_default_model_path(path: &Path) -> bool {
    let default_talker = Path::new("models/qwen-talker-1.7b-base-Q8_0.gguf");
    let default_codec = Path::new("models/qwen-tokenizer-12hz-Q8_0.gguf");
    path == default_talker || path == default_codec
}
```

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| All models exist | Silent, proceed normally |
| Some models missing, user confirms | Download with progress bars, then proceed |
| Some models missing, user declines | Error: "Download cancelled" |
| Network failure during download | Error with HTTP/details message |
| Custom model path is missing | Existing behaviour: error at runner layer |

## Testing

- `test_ensure_default_models_exists` — mock existing files, verify no download
- `test_ensure_default_models_missing_declined` — simulate stdin "n", verify cancellation
- `test_is_default_model_path` — unit test the path detection helper
- Existing tests must continue to pass

## Files Changed

| File | Change |
|------|--------|
| `src/downloader.rs` | Remove dead_code attr, add `ensure_default_models()` |
| `src/main.rs` | Add `is_default_model_path()` + wire into Synth/Inspect handlers |

## Definition of Done

- [ ] `cargo test --release --features ffi` passes all tests
- [ ] Running `synth` with missing default models shows prompt
- [ ] Running `synth` with custom paths does NOT trigger prompt
- [ ] Entering "Y" at prompt downloads models with progress
- [ ] Entering "n" at prompt shows clear cancellation message
