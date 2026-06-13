# Qwen3 TTS Rust App

Local Rust text-to-speech starter app for `Serveurperso/Qwen3-TTS-GGUF`.

## Why qwentts.cpp is used

Qwen3-TTS GGUF is not a normal single LLM GGUF. It needs two GGUF files:

- Talker GGUF: text to 12 Hz codes
- Tokenizer / codec GGUF: codes to 24 kHz mono WAV

This app uses `llama-gguf` to inspect and validate GGUF metadata, and uses `qwentts.cpp` for actual synthesis because qwentts.cpp implements the full TTS pipeline.

## Setup

```bash
cargo run -- setup-script --target cuda > setup.sh
bash setup.sh
```

CPU only:

```bash
cargo run -- setup-script --target cpu > setup.sh
bash setup.sh
```

## Inspect models

```bash
cargo run --release -- inspect \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf
```

## Generate speech

```bash
cargo run --release -- synth \
  --text "你好，這是 Rust 本機語音生成測試。" \
  --lang Chinese \
  --out output.wav \
  --device auto
```

## CustomVoice named speaker

Download a CustomVoice talker first, then run:

```bash
cargo run --release -- synth \
  --talker models/qwen-talker-1.7b-customvoice-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
  --speaker vivian \
  --lang English \
  --text "This is a named speaker demo." \
  --out vivian.wav
```

## Voice cloning

```bash
cargo run --release -- synth \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
  --ref-wav ref.wav \
  --ref-text ref.txt \
  --lang English \
  --text "This is a cloned voice test." \
  --out clone.wav
```

## Config file

Copy:

```bash
cp qwen-tts.toml.example qwen-tts.toml
```

Then run without model flags:

```bash
cargo run --release -- synth --text "Hello" --out out.wav
```
