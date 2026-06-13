# Qwen3 TTS Rust App

Local Rust text-to-speech app for [Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF).

## Why qwentts.cpp is used

Qwen3-TTS GGUF is not a normal single LLM GGUF. It needs two GGUF files:

- **Talker GGUF**: text → 12 Hz speech codes
- **Tokenizer / codec GGUF**: codes → 24 kHz mono WAV

This app uses `llama-gguf` to inspect and validate GGUF metadata, and uses `qwentts.cpp` for actual synthesis because qwentts.cpp implements the full TTS pipeline.

## Quick start

### 1. Build the release binary

```bash
# CLI only
cargo build --release

# CLI + GUI + FFI (qwen.dll support)
cargo build --release --features "gui,ffi"
```

### 2. Get the models

```bash
# Auto-download default models (prompts for confirmation)
cargo run --release -- download

# Or download to a custom directory
cargo run --release -- download --out-dir models
```

The app also auto-downloads missing default models when you run `synth` or `inspect` with default paths (prompts `[Y/n]`).

### 3. Generate speech

```bash
cargo run --release -- synth \
  --text "你好，這是 Rust 本機語音生成測試。" \
  --lang Chinese \
  --out output.wav
```

---

## Voice cloning (複製語音)

Voice cloning synthesises speech in a new voice by referencing a short audio sample.

### How it works

1. Prepare a **reference WAV file** (the voice you want to clone)
2. Prepare a **reference text file** (exact transcription of the WAV)
3. Pass both to `synth` with `--ref-wav` and `--ref-text`

### Audio requirements

| Requirement | Value |
|-------------|-------|
| Format | WAV (PCM) |
| Sample rate | Preferably 16–24 kHz (auto-resampled) |
| Duration | 5–30 seconds of clean speech |
| Channels | Mono strongly preferred |
| Content | Single speaker, clear voice, minimal background noise |

The reference text must be a **word-for-word transcription** in the same language as the reference audio.

### Example

Say you have `speaker_a.wav` (a 10-second recording of person A) and `speaker_a.txt` containing the exact words spoken:

```bash
cargo run --release -- synth \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
  --ref-wav speaker_a.wav \
  --ref-text speaker_a.txt \
  --lang English \
  --text "This is a cloned voice test." \
  --out clone.wav
```

> **Note:** Voice cloning works with the **base** talker model (`qwen-talker-1.7b-base-Q8_0.gguf`), not the CustomVoice variant. The FFI (direct qwen.dll) path currently does not support cloning — the CLI path (qwentts.cpp binary) is used automatically when cloning is requested.

### Prepare the reference files

```bash
# Reference text file (one line, exact transcript)
echo "Hello, this is the original speaker's voice." > speaker_a.txt

# Reference WAV must match the transcript above
# (record with Audacity, ffmpeg, etc.)
ffmpeg -i recording.mp3 -ar 16000 -ac 1 ref.wav
```

---

## CustomVoice named speaker

Download a CustomVoice talker first, then use the `--speaker` flag to pick a named speaker:

```bash
cargo run --release -- synth \
  --talker models/qwen-talker-1.7b-customvoice-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
  --speaker vivian \
  --lang English \
  --text "This is a named speaker demo." \
  --out vivian.wav
```

---

## Speaker modes summary

| Mode | Model | Flags | Description |
|------|-------|-------|-------------|
| Default | `*-base-*.gguf` | (none) | Standard synthesis, no speaker conditioning |
| CustomVoice | `*-customvoice-*.gguf` | `--speaker <name>` | Pick a named speaker from the model |
| Voice cloning | `*-base-*.gguf` | `--ref-wav <file> --ref-text <file>` | Clone a voice from a reference audio clip |

---

## Inspect models

```bash
cargo run --release -- inspect \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf
```

This shows model metadata, GGUF architecture info, and (for CustomVoice models) available speaker names.

---

## GUI desktop app

```bash
# Requires the `gui` feature
cargo run --release --features "gui,ffi" -- gui
```

The GUI provides a visual interface for synthesis with fields for text, language, speaker, voice cloning, and device selection.

---

## Config file

Copy the example and edit paths as needed:

```bash
cp qwen-tts.toml.example qwen-tts.toml
```

Then run without model flags — values are read from `qwen-tts.toml`:

```bash
cargo run --release -- synth --text "Hello" --out out.wav
```

### Config reference

```toml
qwen_tts_bin = "qwentts.cpp/build/qwen-tts"  # qwentts.cpp binary path
talker = "models/qwen-talker-1.7b-base-Q8_0.gguf"  # Talker GGUF model
codec = "models/qwen-tokenizer-12hz-Q8_0.gguf"  # Codec GGUF model
language = "English"  # Default language
device = "auto"  # Compute backend: auto, CPU, CUDA0, Vulkan0, Metal
output_dir = "."  # Output directory for WAV files
hf_repo = "Serveurperso/Qwen3-TTS-GGUF"  # HF repo for downloads
```

---

## GPU acceleration

The app supports three GPU backends via the `--device` flag:

| Backend | GPU | `--device` value | CMake flag |
|---------|-----|-------------------|------------|
| CUDA | NVIDIA | `cuda0` | `-DGGML_CUDA=ON` |
| Vulkan | AMD / Intel / NVIDIA | `vulkan0` | `-DGGML_VULKAN=ON` |
| Metal | Apple Silicon (macOS) | `metal` | `-DGGML_METAL=ON` |
| CPU | Any | `cpu` | (none) |
| Auto | Best available | `auto` | (all flags enabled) |

Select the backend at runtime with `--device`:

```bash
cargo run --release -- synth \
  --text "Hello" --out out.wav \
  --device cuda0   # or vulkan0, metal, cpu, auto
```

Both the CLI (process) and FFI (qwen.dll) paths respect `--device`.

## Build qwentts.cpp (setup script)

Generate a setup script for your platform and backend:

```bash
# CUDA (NVIDIA)
cargo run -- setup-script --target cuda > setup.sh
bash setup.sh

# Vulkan (AMD / Intel)
cargo run -- setup-script --target vulkan > setup.sh
bash setup.sh

# All GPU backends (recommended — single binary, runtime selection)
cargo run -- setup-script --target all > setup.sh
bash setup.sh

# CPU only
cargo run -- setup-script --target cpu > setup.sh
bash setup.sh

# Windows PowerShell (CUDA example)
cargo run -- setup-script --target cuda --powershell > setup.ps1
powershell -File setup.ps1
```

---

## All CLI options

```bash
qwen-tts-app.exe [OPTIONS] [COMMAND]

Commands:
  synth         Generate speech (text → WAV)
  inspect       Inspect talker / codec GGUF metadata
  download      Download model files from Hugging Face Hub
  setup-script  Print setup script for building qwentts.cpp
  gui           Launch the desktop GUI (requires `gui` feature)
  help          Print help

Options:
      --config <FILE>  Optional TOML config file [default: qwen-tts.toml]
  -h, --help           Print help

Synth flags:
      --text <TEXT>          Input text (or pipe via stdin)
      --out <PATH>           Output WAV path [default: out.wav]
      --talker <PATH>        Talker GGUF model path
      --codec <PATH>         Codec GGUF model path
      --lang <LANG>          Language [default: English]
      --speaker <NAME>       Named speaker (CustomVoice models only)
      --instruct <TEXT>      Instruction / style prompt
      --ref-wav <PATH>       Reference WAV for voice cloning
      --ref-text <PATH>      Reference transcript for voice cloning
      --device <DEVICE>      Compute backend [default: auto]
```
