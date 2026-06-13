# Lessons Learned

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
