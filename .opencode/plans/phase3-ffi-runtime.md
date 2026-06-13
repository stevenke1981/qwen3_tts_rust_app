## Plan: Phase 3 — FFI Runtime
**Goal:** Replace process-based qwen-tts binary invocation with direct FFI calls to qwen.dll (shared library).
**Complexity:** L4

### Strategy
- Dynamic library loading via `libloading` (no build-time link dependency)
- Auto-detect DLL at runtime → FFI; fallback to process-based runner
- Safe Rust wrapper around raw C ABI

### Files to create
1. `src/qwen_ffi.rs` — Raw FFI types + QwenLibrary safe wrapper (~300 lines)

### Files to modify
2. `Cargo.toml` — Add `libloading` dep, `ffi` feature
3. `src/qwentts_cli.rs` — Add `Synthesizer` trait, `QwenFfiRunner` impl
4. `src/main.rs` — Auto-detect DLL, use FFI when available
5. `src/gui.rs` — Use FFI runner when available

### Sub-tasks
1. [ ] Write `src/qwen_ffi.rs` — raw FFI types (repr(C) structs, fn ptrs)
2. [ ] Write `src/qwen_ffi.rs` — QwenLibrary (load DLL, safe wrappers)
3. [ ] Update `Cargo.toml` — libloading dep, ffi feature
4. [ ] Update `src/qwentts_cli.rs` — Synthesizer trait
5. [ ] Update `src/qwentts_cli.rs` — QwenFfiRunner
6. [ ] Update `src/main.rs` — auto-detect DLL
7. [ ] Update `src/gui.rs` — use Synthesizer trait
8. [ ] Build & test (cargo check + cargo check --features ffi)
9. [ ] git commit

### Risks
| Risk | Mitigation |
|------|------------|
| C struct layout mismatch | repr(C), match types exactly, verify with test |
| DLL not available at runtime | Graceful fallback to process-based runner |
| Windows ABI quirks | Use extern "C", test on Windows explicitly |

### Definition of Done
- [ ] `cargo check --all-features` succeeds
- [ ] `cargo test` passes (all 12+ tests)
- [ ] FFI module compiles without a DLL present
- [ ] Process-based fallback still works when DLL absent
- [ ] git commit created
