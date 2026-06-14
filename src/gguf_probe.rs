use anyhow::{Context, Result};
use llama_gguf::{default_backend, GgufFile};
use std::path::Path;

/// Key metadata fields to extract from a GGUF file
const INTERESTING_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    "general.description",
    "general.size_label",
    "general.file_type",
    "general.alignment",
    "general.version",
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.rope.freq_base",
    "llama.rope.scaling.type",
    "tokenizer.ggml.model",
    "qwen3tts.type",
    "qwen3tts.audio.channels",
    "qwen3tts.audio.sample_rate",
    "qwen3tts.audio.codec_dim",
    "qwen3tts.talker.num_speakers",
    "qwen3tts.talker.speakers",
    "qwen3tts.codec.codebooks",
];

fn extract_metadata(file: &GgufFile) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for key in INTERESTING_KEYS {
        if let Some(val) = file.data.get_string(key) {
            pairs.push((key.to_string(), val.into()));
        } else if let Some(val) = file.data.get_u32(key) {
            pairs.push((key.to_string(), val.to_string()));
        } else if let Some(val) = file.data.get_u64(key) {
            pairs.push((key.to_string(), val.to_string()));
        } else if let Some(val) = file.data.get_f32(key) {
            pairs.push((key.to_string(), format!("{val:.4}")));
        } else if let Some(val) = file.data.get_bool(key) {
            pairs.push((key.to_string(), val.to_string()));
        }
    }
    pairs
}

pub fn inspect_pair(talker: &Path, codec: &Path) -> Result<String> {
    let backend = default_backend();
    let talker_file = GgufFile::open(talker)
        .with_context(|| format!("failed to open talker GGUF: {}", talker.display()))?;
    let codec_file = GgufFile::open(codec)
        .with_context(|| format!("failed to open codec GGUF: {}", codec.display()))?;

    let mut output = String::new();
    output.push_str(&format!(
        "═══════════════════════════════════════════\n\
         GGUF Metadata Inspection\n\
         Backend: {}\n\
         ═══════════════════════════════════════════\n\n",
        backend.name()
    ));

    // --- Talker ---
    output.push_str(&format!(
        "── Talker ──────────────────────────────\n  Path: {}\n",
        talker.display()
    ));
    let talker_meta = extract_metadata(&talker_file);
    if talker_meta.is_empty() {
        output.push_str("  (no recognized metadata keys)\n");
    } else {
        for (k, v) in &talker_meta {
            output.push_str(&format!("  {k}: {v}\n"));
        }
    }

    // Show metadata count
    output.push_str(&format!(
        "\n  Metadata entries: {}\n",
        talker_file.data.metadata.len()
    ));

    output.push_str("\n");

    // --- Codec ---
    output.push_str(&format!(
        "── Codec ───────────────────────────────\n  Path: {}\n",
        codec.display()
    ));
    let codec_meta = extract_metadata(&codec_file);
    if codec_meta.is_empty() {
        output.push_str("  (no recognized metadata keys)\n");
    } else {
        for (k, v) in &codec_meta {
            output.push_str(&format!("  {k}: {v}\n"));
        }
    }

    output.push_str(&format!(
        "\n  Metadata entries: {}\n",
        codec_file.data.metadata.len()
    ));

    // --- Validation ---
    output.push_str("\n── Validation ───────────────────────────\n");

    let talker_arch = talker_file
        .data
        .get_string("general.architecture")
        .unwrap_or("unknown");
    let codec_arch = codec_file
        .data
        .get_string("general.architecture")
        .unwrap_or("unknown");

    let talker_has_tts_keys = talker_file.data.get_string("qwen3tts.type").is_some()
        || talker_file
            .data
            .get_u32("qwen3tts.talker.num_speakers")
            .is_some()
        || talker_file
            .data
            .get_u32("qwen3tts.audio.channels")
            .is_some();

    let codec_has_codec_keys = codec_file.data.get_string("qwen3tts.type").is_some()
        || codec_file
            .data
            .get_u32("qwen3tts.codec.codebooks")
            .is_some()
        || codec_file
            .data
            .get_u32("qwen3tts.audio.codec_dim")
            .is_some();

    if talker_arch.contains("qwen3") {
        if talker_has_tts_keys {
            output.push_str(&format!(
                "  ✅ Talker architecture '{talker_arch}' with TTS metadata — confirmed Qwen3-TTS.\n"
            ));
        } else {
            output.push_str(&format!(
                "  ⚠ Talker architecture '{talker_arch}' but no TTS metadata keys found.\n\
                  ⚠ This may be a plain Qwen3 LLM GGUF, not a TTS model.\n"
            ));
        }
    } else {
        output.push_str(&format!(
            "  ⚠ Talker architecture '{talker_arch}' — may not be Qwen3-TTS.\n"
        ));
    }

    if codec_arch.contains("qwen3") || codec_arch.contains("audio") || codec_arch.contains("codec")
    {
        if codec_has_codec_keys {
            output.push_str(&format!(
                "  ✅ Codec architecture '{codec_arch}' with codec metadata — confirmed TTS codec.\n"
            ));
        } else {
            output.push_str(&format!(
                "  ⚠ Codec architecture '{codec_arch}' but no TTS codec metadata keys found.\n\
                  ⚠ This may be a plain Qwen3 LLM GGUF or a non-TTS audio model.\n"
            ));
        }
    } else {
        output.push_str(&format!(
            "  ⚠ Codec architecture '{codec_arch}' — may not be correct.\n"
        ));
    }

    // Add TTS type classification if available
    if let Some(tts_type) = talker_file.data.get_string("qwen3tts.type") {
        output.push_str(&format!("  ℹ Talker TTS type: {tts_type}\n"));
    }
    if let Some(tts_type) = codec_file.data.get_string("qwen3tts.type") {
        output.push_str(&format!("  ℹ Codec TTS type: {tts_type}\n"));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interesting_keys_non_empty() {
        assert!(!INTERESTING_KEYS.is_empty());
        assert!(INTERESTING_KEYS.contains(&"general.architecture"));
    }
}
