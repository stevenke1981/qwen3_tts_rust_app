# GPU Acceleration — Design Spec

**Date:** 2026-06-14
**Status:** Design — approved, awaiting implementation plan

## Goal

Add GPU acceleration support for qwen3-tts-rust-app across three backends:

1. **CUDA** — NVIDIA GPUs
2. **Vulkan** — AMD / Intel / NVIDIA (cross-platform)
3. **Metal** — Apple Silicon (macOS)

The app already supports `--device cuda0|vulkan0|metal|cpu|auto` at the CLI level and passes `GGML_BACKEND` env var to the `qwen-tts` process. The missing pieces are:

- FFI path (`qwen.dll`) cannot select GPU backend at runtime
- Build scripts do not emit correct CMake flags for each backend
- `BuildTarget` enum lacks `Metal`
- No CI coverage for GPU-backed builds

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         qwen-tts-app.exe            │
                    │                                     │
                    │  --device cuda0/vulkan0/metal/cpu   │
                    │         │                           │
                    │         ▼                           │
                    │  ┌──────────────┐  ┌─────────────┐ │
                    │  │ CLI path     │  │ FFI path    │ │
                    │  │ (process)    │  │ (qwen.dll)  │ │
                    │  │ GGML_BACKEND │  │ params.back │ │
                    │  │ env var      │  │ end field   │ │
                    │  └──────┬───────┘  └─────┬───────┘ │
                    └─────────┼────────────────┼─────────┘
                              │                │
                              ▼                ▼
                    ┌──────────────┐  ┌──────────────┐
                    │  qwen-tts    │  │  qwen.dll    │
                    │  (spawned)   │  │  (GGML_ALL)  │
                    └──────┬───────┘  └──────┬───────┘
                           │                  │
                           ▼                  ▼
                    ┌───────────────────────────────────┐
                    │       ggml backends               │
                    │  CUDA · Vulkan · Metal · CPU      │
                    └───────────────────────────────────┘
```

## Changes

### 1. qwentts.cpp — ABI v2 → v3

**File:** `src/qwen.h` (in the qwentts.cpp repo)

Add two fields at the end of `struct qt_init_params`:

```c
#define QT_ABI_VERSION 3

struct qt_init_params {
    int          abi_version;     // 3
    const char * talker_path;
    const char * codec_path;
    bool         use_fa;
    bool         clamp_fp16;
    // ── ABI v3 ──
    const char * backend;         // NULL = auto
    int          n_gpu_layers;    // -1 = all, 0 = CPU
};
```

**File:** `src/qwen.cpp`

`qt_init_default_params` — zero-init new fields:

```cpp
void qt_init_default_params(struct qt_init_params * p) {
    p->abi_version = QT_ABI_VERSION;   // 3
    p->talker_path = nullptr;
    p->codec_path  = nullptr;
    p->use_fa      = true;
    p->clamp_fp16  = false;
    p->backend      = nullptr;          // new
    p->n_gpu_layers = -1;               // new
}
```

`qt_init` — read `backend` field and set env var before `backend_init`:

```cpp
// In qt_init(), before backend_init("Talker"):
if (params->abi_version >= 3 && params->backend) {
#ifdef _WIN32
    SetEnvironmentVariableA("GGML_BACKEND", params->backend);
#else
    setenv("GGML_BACKEND", params->backend, 1);
#endif
}

q->bp = backend_init("Talker");
```

**Backward compatibility:** The existing ABI v2 struct has the same memory layout
for the first 5 fields. The new fields occupy memory that was previously
uninitialized tail padding — but since `qt_init_default_params` is always called
before `qt_init`, v2 callers that skip zero-initialization still work because
`abi_version` check (`params->abi_version > QT_ABI_VERSION`) rejects structs
from newer bindings, never the other way around.

The `n_gpu_layers` field is added for future use. The current `backend_init`
does not read `n_gpu_layers` — ggml handles GPU layer placement internally.
A follow-up could pass it to a `ggml_backend_set_n_gpu_layers` call.

### 2. Rust FFI — `qwen_ffi.rs`

**`QtInitParamsRaw`** — match the new C struct:

```rust
pub(crate) struct QtInitParamsRaw {
    abi_version: i32,
    talker_path: *const i8,
    codec_path: *const i8,
    use_fa: bool,
    clamp_fp16: bool,
    // ABI v3
    backend: *const i8,
    n_gpu_layers: i32,
}
```

**`QwenLibrary::init()`** — accept `backend` and `n_gpu_layers`:

```rust
pub fn init(
    &self,
    talker_path: &str,
    codec_path: &str,
    use_fa: bool,
    clamp_fp16: bool,
    backend: Option<&str>,
    n_gpu_layers: i32,
) -> Result<*mut QtContext, QwenFfiError>
```

The `backend` string is converted to a `CString` and passed via the struct
field (null if `None`). No env var manipulation needed — the C lib handles it.

**`QwenFfiRunner::synthesize()`** — pass `backend` from `req.ggml_backend`:

```rust
let backend_str = req.ggml_backend.as_deref();
let ctx = self.lib.init(
    &talker_str, &codec_str, true, false,
    backend_str, -1,  // backend, n_gpu_layers = all
)?;
```

### 3. Rust Backend Routing — `main.rs`

**`Device` enum** — already has `Auto | Cpu | Cuda0 | Vulkan0 | Metal`; no change.

**`Device::backend_str()`** — rename `ggml_backend_env()` to `backend_str()` since it's used for both paths:

```rust
fn backend_str(&self) -> Option<&'static str> {
    match self {
        Device::Auto => None,
        Device::Cpu => Some("CPU"),
        Device::Cuda0 => Some("CUDA0"),
        Device::Vulkan0 => Some("Vulkan0"),
        Device::Metal => Some("Metal"),
    }
}
```

**Synth handler routing** — both paths now use the same backend string:

```
CLI path:   cmd.env("GGML_BACKEND", backend_str)  ← already works
FFI path:   lib.init(..., backend_str, -1)          ← new
```

### 4. Build Scripts — `main.rs`

**`BuildTarget`** — add `Metal`:

```rust
enum BuildTarget { Cpu, Cuda, Vulkan, Metal, All }
```

**CMake flag mapping:**

| Target | CMake Flags |
|--------|------------|
| `cpu` | (no GPU flags — ggml auto-detects) |
| `cuda` | `-DGGML_CUDA=ON` |
| `vulkan` | `-DGGML_VULKAN=ON` |
| `metal` | `-DGGML_METAL=ON` |
| `all` | `-DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_METAL=ON` |

In the PowerShell setup script (`print_setup_script_powershell`), replace the
current hardcoded CUDA-only CMake line with a backend-aware switch:

```powershell
$ggml_flags = switch ("$backend") {
    "cuda"   { "-DGGML_CUDA=ON" }
    "vulkan" { "-DGGML_VULKAN=ON" }
    "metal"  { "-DGGML_METAL=ON" }
    "all"    { "-DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_METAL=ON" }
    default  { "" }
}
cmake .. -DQWEN_SHARED=ON $ggml_flags
```

In the bash setup script, maintain the same approach.

### 5. CI/CD — `.github/workflows/release.yml`

Update the `Build qwen.dll` step to enable all GPU backends:

```yaml
- name: Build qwen.dll
  working-directory: qwentts.cpp
  run: |
    cmake -B build -DQWEN_SHARED=ON `
      -DGGML_CUDA=ON -DGGML_VULKAN=ON
    cmake --build build --config Release
```

Note: `-DGGML_METAL=ON` is omitted from the Windows CI build since Metal is
macOS-only. A separate `macos-latest` job could be added for Metal builds.

## Files Changed

### qwentts.cpp repo (fork: stevenke1981/qwentts.cpp)

| File | Change |
|------|--------|
| `src/qwen.h` | Bump `QT_ABI_VERSION` 2→3, add `backend`, `n_gpu_layers` fields |
| `src/qwen.cpp` | Update `qt_init_default_params`, set `GGML_BACKEND` in `qt_init` |

### Rust repo (stevenke1981/qwen3_tts_rust_app)

| File | Change |
|------|--------|
| `src/qwen_ffi.rs` | Update `QtInitParamsRaw`, `QwenLibrary::init()` signature |
| `src/main.rs` | Update `create_synth_runner`, `BuildTarget`, scripts |
| `.github/workflows/release.yml` | Enable GPU backends in CMake |
| `README.md` | Document GPU build instructions and device flags |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| ABI v3 struct may mismatch between old binding + new lib | `abi_version` check in `qt_init` rejects newer structs |
| `GGML_BACKEND` env var is process-global; setting it before FFI `qt_init` affects other threads | The FFI path is single-threaded at init time; CPU path is unaffected |
| GPU backends not available in CI (no NVIDIA GPU on GitHub runners) | `cmake` enables the backend code; `backend_init` falls back to CPU at runtime if no GPU is present |
| `ggml_backend_init_by_name` may not find "CUDA0" etc. | The runtime logs a clear error with available devices |

## Definition of Done

- [ ] qwentts.cpp: ABI v3 struct + `qt_init_default_params` update committed to fork
- [ ] qwen_ffi.rs: new struct + updated `init()` signature
- [ ] main.rs: backend routing for FFI path, `BuildTarget::Metal`, setup scripts
- [ ] `.github/workflows/release.yml`: GPU flags in CMake
- [ ] `cargo build --release --features "gui,ffi"` compiles clean
- [ ] `cargo test --release --features "gui,ffi"` passes
- [ ] `cargo run -- setup-script --target all` prints correct CMake flags
- [ ] README updated with GPU build instructions

## Assumptions

- The qwentts.cpp fork lives at `github.com/stevenke1981/qwentts.cpp`
- The CI runner's pre-installed CUDA/Vulkan SDKs are sufficient for compilation
- macOS Metal CI is not required for initial release (manual build only)
