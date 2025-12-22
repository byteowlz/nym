# Install script for nym on Windows
# Run with: powershell -ExecutionPolicy Bypass -File scripts\install-nym.ps1

$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=========================================="
Write-Host "  nym - PII Anonymization CLI Installer"
Write-Host "=========================================="
Write-Host ""

# Check if NER is wanted
Write-Host "Do you want NER support for detecting names and addresses?"
Write-Host "NER uses machine learning and requires downloading a ~50MB model."
Write-Host ""
Write-Host "  1) No  - Regex-only detection (fast, no model download)"
Write-Host "  2) Yes - Include NER support (more accurate for names/addresses)"
Write-Host ""

$NER_CHOICE = Read-Host "Enter choice [1]"
if ([string]::IsNullOrEmpty($NER_CHOICE)) { $NER_CHOICE = "1" }

$FEATURES = @()

if ($NER_CHOICE -eq "2") {
    Write-Host ""
    Write-Host "NER enabled. Select hardware acceleration:"
    Write-Host ""
    Write-Host "  1) CPU only - works everywhere"
    Write-Host "  2) DirectML (Windows GPU) - uses DirectX 12"
    Write-Host "  3) CUDA (NVIDIA GPU) - requires CUDA toolkit"
    Write-Host "  4) TensorRT (NVIDIA optimized) - requires TensorRT"
    Write-Host "  5) OpenVINO (Intel) - Intel CPU/GPU optimization"
    Write-Host ""
    Write-Host "Recommended for Windows: DirectML (2) or CPU (1)"
    Write-Host ""

    $HW_CHOICE = Read-Host "Enter choice [1]"
    if ([string]::IsNullOrEmpty($HW_CHOICE)) { $HW_CHOICE = "1" }

    switch ($HW_CHOICE) {
        "1" { $FEATURES = @("--features", "ner") }
        "2" { $FEATURES = @("--features", "ner-directml") }
        "3" { $FEATURES = @("--features", "ner-cuda") }
        "4" { $FEATURES = @("--features", "ner-tensorrt") }
        "5" { $FEATURES = @("--features", "ner-openvino") }
        default { 
            Write-Host "Invalid choice. Using CPU-only NER."
            $FEATURES = @("--features", "ner")
        }
    }
    $ENABLE_NER = $true
} else {
    $ENABLE_NER = $false
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Building nym..."
Write-Host "=========================================="
Write-Host ""

if ($FEATURES.Count -gt 0) {
    Write-Host "Features: $($FEATURES[1])"
    cargo build --release @FEATURES --manifest-path "$ROOT_DIR\Cargo.toml"
} else {
    Write-Host "Features: none (regex-only)"
    cargo build --release --manifest-path "$ROOT_DIR\Cargo.toml"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=========================================="
Write-Host "  Installing nym..."
Write-Host "=========================================="
Write-Host ""

if ($FEATURES.Count -gt 0) {
    cargo install --path "$ROOT_DIR" @FEATURES --force
} else {
    cargo install --path "$ROOT_DIR" --force
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=========================================="
Write-Host "  Installation complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Binary installed to: $env:USERPROFILE\.cargo\bin\nym.exe"
Write-Host ""
Write-Host "Quick start:"
Write-Host "  nym detect file.txt          # Detect PII in a file"
Write-Host "  nym anon file.txt            # Anonymize PII in a file"
Write-Host "  nym anon -k keys.jsonl file  # Anonymize with reversible key file"
Write-Host "  nym deanon -k keys.jsonl     # Restore original PII"
Write-Host ""

if ($ENABLE_NER) {
    Write-Host "NER support enabled. Use --ner flag to detect names/addresses:"
    Write-Host "  nym detect --ner file.txt"
    Write-Host "  nym anon --ner file.txt"
    Write-Host ""
    Write-Host "The NER model will be downloaded on first use (~50MB)."
}
