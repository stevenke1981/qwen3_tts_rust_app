<#
.SYNOPSIS
    Build and package Qwen3-TTS Rust App for Windows distribution.
.DESCRIPTION
    Compiles the release binary, gathers required DLLs, config template,
    and verification scripts into a portable .zip archive.

    Prerequisites:
      - Protobuf compiler (protoc) path set in $env:PROTOC
      - 7-Zip installed at the configured path (adjust $7zPath below)
      - Optional: qwen.dll, ggml*.dll from a qwentts.cpp build

    Usage:
      .\scripts\package.ps1
#>

$ErrorActionPreference = "Stop"

$rootDir    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$distDir    = Join-Path $rootDir "dist"
$stageDir   = Join-Path $distDir "stage"
$releaseDir = Join-Path $rootDir "target\release"

# ── Config ──
$archiveName  = "qwen3-tts-app-v0.1.0.zip"
$7zPath       = "C:\Program Files\NVIDIA Corporation\NVIDIA app\7z.exe"
$features     = "gui,ffi"
$protocPath   = Join-Path $rootDir ".local\bin\protoc.exe"

Write-Host "╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Qwen3-TTS Windows Release Packager          ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Build ──
Write-Host "Step 1: Building release binary..." -ForegroundColor Yellow
$env:PROTOC = $protocPath
Push-Location $rootDir
cargo build --release --features $features 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Release build complete" -ForegroundColor Green
Pop-Location

# ── Step 2: Stage files ──
Write-Host "Step 2: Staging files..." -ForegroundColor Yellow
Remove-Item $stageDir -Recurse -Force -ErrorAction SilentlyContinue

$dirs = @("models", "output", "scripts")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path (Join-Path $stageDir $d) -Force | Out-Null
}

# Main binary
Copy-Item (Join-Path $releaseDir "qwen-tts-app.exe") $stageDir

# Runtime DLLs (qwentts.cpp build artifacts) — optional
$dllCandidates = @("qwen.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")
foreach ($dll in $dllCandidates) {
    $src = Join-Path $rootDir $dll
    if (Test-Path $src) {
        Copy-Item $src $stageDir
    }
}

# Config template + README
$configSrc = Join-Path $rootDir "qwen-tts.toml.example"
if (Test-Path $configSrc) { Copy-Item $configSrc $stageDir }
$readmeSrc = Join-Path $rootDir "README.md"
if (Test-Path $readmeSrc) { Copy-Item $readmeSrc $stageDir }

# Scripts
$scriptsSrc = Join-Path $rootDir "scripts"
Get-ChildItem "$scriptsSrc\*.ps1", "$scriptsSrc\*.sh" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $stageDir "scripts") }

# ── Step 3: Create archive ──
Write-Host "Step 3: Creating archive..." -ForegroundColor Yellow
$zipPath = Join-Path $distDir $archiveName
Remove-Item $zipPath -ErrorAction SilentlyContinue

Push-Location $stageDir
if (Test-Path $7zPath) {
    & $7zPath a -tzip $zipPath * -mx=9 -bsp0 2>&1 | Out-Null
} else {
    # Fallback to .NET compression (slower, no progress)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(".", $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
}
Pop-Location

# ── Step 4: Cleanup + report ──
Remove-Item $stageDir -Recurse -Force -ErrorAction SilentlyContinue

$zipInfo = Get-Item $zipPath
$sizeMB = [math]::Round($zipInfo.Length / 1MB, 2)
Write-Host "  ✅ Archive created: $archiveName ($sizeMB MB)" -ForegroundColor Green

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Package complete                             ║" -ForegroundColor Green
Write-Host "║  $zipPath" -ForegroundColor White
Write-Host "╚═══════════════════════════════════════════════╝" -ForegroundColor Cyan
