<#
.SYNOPSIS
    Qwen3-TTS Quick Smoke Test — verifies a WAV artifact is produced.
.DESCRIPTION
    Checks prerequisites (models, binary/DLL), runs a short Chinese prompt,
    and validates output file size/header.
#>

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Qwen3-TTS Smoke Test" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check required model files ──
$modelsDir = Join-Path $rootDir "models"
$talkerExpected = "qwen-talker-1.7b-base-Q8_0.gguf"
$codecExpected  = "qwen-tokenizer-12hz-Q8_0.gguf"
$talkerPath     = Join-Path $modelsDir $talkerExpected
$codecPath      = Join-Path $modelsDir $codecExpected

Write-Host "Step 1: Checking model files..." -ForegroundColor Yellow
$missing = @()
if (-not (Test-Path $talkerPath)) { $missing += $talkerExpected }
if (-not (Test-Path $codecPath))  { $missing += $codecExpected }

if ($missing.Count -gt 0) {
    Write-Host "⚠ Missing model files:" -ForegroundColor Red
    foreach ($f in $missing) { Write-Host "  - $f" }
    Write-Host ""
    Write-Host "Run: cargo run -- download --out-dir models" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✅ Talker: $talkerExpected" -ForegroundColor Green
Write-Host "  ✅ Codec:  $codecExpected" -ForegroundColor Green

# ── Step 2: Check qwen-tts binary ──
Write-Host "Step 2: Checking qwen-tts binary..." -ForegroundColor Yellow
$binCandidates = @(
    Join-Path $rootDir "qwentts.cpp/build/Release/qwen-tts.exe"
    Join-Path $rootDir "qwentts.cpp/build/qwen-tts.exe"
    Join-Path $rootDir "qwentts.cpp/build/bin/qwen-tts.exe"
)
$qwenBin = $null
foreach ($c in $binCandidates) {
    if (Test-Path $c) {
        $qwenBin = $c
        break
    }
}

$qwenDll = Join-Path $rootDir "qwen.dll"
if ($null -eq $qwenBin -and (-not (Test-Path $qwenDll))) {
    Write-Host "  ⚠ Neither qwen-tts.exe nor qwen.dll found." -ForegroundColor Red
    Write-Host "  Build: cmake --build qwentts.cpp/build --config Release" -ForegroundColor Yellow
    Write-Host "  Or place qwen.dll in project root." -ForegroundColor Yellow
    exit 1
}
if ($null -ne $qwenBin) {
    Write-Host "  ✅ Binary: $qwenBin" -ForegroundColor Green
} else {
    Write-Host "  ✅ FFI library: qwen.dll" -ForegroundColor Green
}

# ── Step 3: Run a short synthesis ──
Write-Host "Step 3: Running synthesis..." -ForegroundColor Yellow
$outPath = Join-Path $rootDir "output"
if (-not (Test-Path $outPath)) { New-Item -ItemType Directory -Path $outPath -Force | Out-Null }

$outFile = Join-Path $outPath "smoke_test_output.wav"
$prompt = "你好，世界！这是Qwen3 TTS的测试。"

# Pick the runner based on what's available
if ($null -ne $qwenBin) {
    # Process runner
    Write-Host "  Using process runner: $qwenBin" -ForegroundColor Gray
    & $qwenBin `
        --model $talkerPath `
        --codec $codecPath `
        --lang Chinese `
        -o $outFile `
        --stdin 2>&1 | Out-Null
    # Send text via stdin (PowerShell pipeline)
    $prompt | & $qwenBin `
        --model $talkerPath `
        --codec $codecPath `
        --lang Chinese `
        -o $outFile `
        --stdin 2>&1
} else {
    # Use the Rust app with FFI (if built with gui,ffi features)
    Write-Host "  Using cargo run -- synth (FFI)" -ForegroundColor Gray
    cargo run --release --features "ffi" -- synth `
        --text $prompt `
        --talker $talkerPath `
        --codec $codecPath `
        --lang Chinese `
        --out $outFile 2>&1
}

# ── Step 4: Validate output WAV ──
Write-Host "Step 4: Validating output..." -ForegroundColor Yellow
if (-not (Test-Path $outFile)) {
    Write-Host "  ❌ Output file not found: $outFile" -ForegroundColor Red
    exit 1
}

$fileInfo = Get-Item $outFile
$sizeBytes = $fileInfo.Length
Write-Host "  File size: $sizeBytes bytes" -ForegroundColor Gray

if ($sizeBytes -lt 5000) {
    Write-Host "  ❌ Output file too small ($sizeBytes bytes) — likely corrupt." -ForegroundColor Red
    exit 1
}

# Check WAV header (first 4 bytes should be "RIFF")
$stream = [System.IO.File]::OpenRead($outFile)
$reader = New-Object System.IO.BinaryReader($stream)
$riff = [char[]]@($reader.ReadBytes(4)) -join ''
$reader.Close()
$stream.Close()

if ($riff -ne "RIFF") {
    Write-Host "  ❌ Not a valid WAV file (header: $riff)" -ForegroundColor Red
    exit 1
}

Write-Host "  ✅ WAV header valid (RIFF)" -ForegroundColor Green
Write-Host "  ✅ Output: $outFile ($([math]::Round($sizeBytes / 1KB, 1)) KB)" -ForegroundColor Green

# ── Cleanup ──
Remove-Item $outFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Smoke test PASSED" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
exit 0
