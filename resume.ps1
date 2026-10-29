# Windows PowerShell Resume Script for Modern Transformer
# Usage: .\resume.ps1 [-Checkpoint <path>] [-Steps <int>]
param (
    [string]$Checkpoint = "checkpoints/modern_production_v2_best.pt",
    [int]$Steps = 200,
    [float]$LR = 0.0001
)

$env:OPENBLAS_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

$trainScript = Join-Path $PSScriptRoot "tiny_llm_from_scratch\train_modern.py"

Write-Host "[RESUME] Resuming training from checkpoint $Checkpoint for $Steps steps..." -ForegroundColor Cyan
& $venvPython $trainScript --resume $Checkpoint --steps $Steps --lr $LR
