#!/usr/bin/env bash
set -euo pipefail

git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
cd qwentts.cpp
./buildcuda.sh
cd ..

mkdir -p models
huggingface-cli download Serveurperso/Qwen3-TTS-GGUF \
  qwen-talker-1.7b-base-Q8_0.gguf \
  qwen-tokenizer-12hz-Q8_0.gguf \
  --local-dir models

cargo run --release -- inspect \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf

cargo run --release -- synth \
  --text "Hello from Rust and Qwen3 TTS." \
  --out out.wav \
  --device cuda0
