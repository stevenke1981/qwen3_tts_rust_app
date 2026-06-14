# Lessons Learned

---
## Lesson #8 — 2026-06-14
**Trigger:** GUI app flashed and exited silently on double-click; CLI commands produced no output.
**Rule:** `#![windows_subsystem = "windows"]` suppresses the console for ALL modes (both GUI and CLI). Instead of using this attribute, compile as console-subsystem and let GUI mode call `ShowWindow(GetConsoleWindow(), SW_HIDE)` on startup. This way CLI commands get stdout/stderr, and GUI mode still hides the console window.
**Source:** Fix double-click crash and CLI output
---
## Lesson #1 — 2026-06-14
**Trigger:** Rust build failed because `protoc` was missing when compiling `llama-gguf` crate.
**Rule:** Before running `cargo build`, check if `llama-gguf` (or similar protobuf-dependent crates) are in the dependency tree. If so, ensure `protoc` is installed first. On Windows without admin rights, download the binary directly from GitHub and set `$env:PROTOC` to its path.
**Source:** Build release and qwen.dll
---
## Lesson #2 — 2026-06-14
**Trigger:** Building `qwentts.cpp` required a C++ compiler; initial check showed no MSVC.
**Rule:** Use `Get-ChildItem -Recurse -Filter "cl.exe"` under VS install directories to find MSVC compiler (it may be installed but not in PATH). Then use `vcvars64.bat` via `cmd /c "... && set"` to capture environment variables into PowerShell.
**Source:** Build release and qwen.dll
---
## Lesson #3 — 2026-06-14
**Trigger:** Chocolatey package installs failed due to missing admin rights.
**Rule:** On Windows without admin, download pre-built binary archives from GitHub Releases directly using `Invoke-WebRequest` and extract with `Expand-Archive`. This avoids the admin requirement entirely.
**Source:** Build release and qwen.dll
---
## Lesson #4 — 2026-06-14
**Trigger:** Test hung when `prompt_yes_no()` tried to read stdin in non-interactive test environment.
**Rule:** Use `std::io::Stdin::is_terminal()` (stabilized in Rust 1.70) to detect interactive TTY before prompting user. In tests/non-interactive mode, default to `false` (decline) instead of blocking on stdin read.
**Source:** Auto-download models feature
---
## Lesson #5 — 2026-06-14
**Trigger:** GitHub Actions workflow needed to build qwen.dll — DLLs end up in different subdirectories depending on CMake target.
**Rule:** After `cmake --build build --config Release`, check both `build/Release/` (for the main target like `qwen.dll`) and `build/src/Release/` (for dependency DLLs like ggml*.dll). Always add a diagnostic `Get-ChildItem -Recurse *.dll` step to verify locations before staging.
**Source:** GitHub Actions workflow for release build
---
## Lesson #6 — 2026-06-14
**Trigger:** After refactoring PowerShell template to use `$ggml_flags = switch ("{backend}")`, the old `backend_flag` format arg was removed from the `println!` call but a stale comment still referenced `{backend_flag}`, causing a compile error for unused variable.
**Rule:** After changing template variable names in format strings, run a full `cargo check --features "ffi"` (not just `cargo check` without features) to catch stale variable references anywhere in the same file.
**Source:** GPU acceleration Task 6 — fix setup scripts
---
## Lesson #7 — 2026-06-14
**Trigger:** Added `n_gpu_layers` field to `QwenTtsRequest` but forgot to update the `QwenTtsApp::start_generation()` constructor in `gui.rs`, causing a compile error.
**Rule:** When adding a required field to a struct used across multiple files (main.rs, gui.rs, test modules), search all constructors with `grep` before making the change, then fix them all in one pass before the first compile check.
**Source:** n_gpu_layers CLI flag implementation
