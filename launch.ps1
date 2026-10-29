# Windows PowerShell Launcher for Modern Tiny LLM Chat & Reasoning Assistant
# Usage: .\launch.ps1 [-Checkpoint <path>] [-Device <cpu|cuda|mps|auto>]
param (
    [string]$Checkpoint = "",
    [string]$Device = "auto"
)

# Set Windows OpenBLAS thread guards to prevent memory allocation retry crashes
$env:OPENBLAS_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Safe UTF-8 Console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

$chatScript = Join-Path $PSScriptRoot "chat.py"
if (-not (Test-Path $chatScript)) {
    $chatScript = Join-Path $PSScriptRoot "tiny_llm_from_scratch\chat.py"
}

$argsList = @($chatScript)
if ($Checkpoint -ne "") {
    $argsList += "--checkpoint"
    $argsList += $Checkpoint
}
if ($Device -ne "") {
    $argsList += "--device"
    $argsList += $Device
}

& $venvPython @argsList

