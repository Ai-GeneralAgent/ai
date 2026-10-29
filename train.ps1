# Windows PowerShell Training Runner for Modern Transformer
# Usage: .\train.ps1 [-Steps <int>] [-BatchSize <int>] [-GradAccum <int>] [-LR <float>] [-Data <path>]
param (
    [int]$Steps = 100,
    [int]$BatchSize = 4,
    [int]$GradAccum = 2,
    [float]$LR = 0.0003,
    [string]$Data = "data_knowledge_corpus.txt",
    [string]$Device = "auto"
)

# Set Windows OpenBLAS thread guards
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

Write-Host "[TRAIN] Launching training on $Data for $Steps steps..." -ForegroundColor Cyan
& $venvPython $trainScript --steps $Steps --batch-size $BatchSize --grad-accum $GradAccum --lr $LR --data $Data --device $Device
