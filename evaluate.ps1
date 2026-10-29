# Windows PowerShell Evaluation Benchmark Runner
# Usage: .\evaluate.ps1

$env:OPENBLAS_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

$evalScript = Join-Path $PSScriptRoot "tiny_llm_from_scratch\compare_models.py"

Write-Host "[EVALUATE] Running empirical 6-task standardized evaluation benchmark..." -ForegroundColor Cyan
& $venvPython $evalScript

