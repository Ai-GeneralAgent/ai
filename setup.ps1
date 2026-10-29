# Windows PowerShell Setup Script for Modern Tiny LLM & Agent
# Usage: .\setup.ps1

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Modern Tiny LLM & Autonomous Agent - Windows Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python installation
$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python 3 was not found in PATH. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

$pyVersion = & $pythonCmd --version
Write-Host "[INFO] Detected Python: $pyVersion" -ForegroundColor Green

# 2. Virtual Environment check/creation
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creating virtual environment at .venv..." -ForegroundColor Yellow
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[INFO] Virtual environment ready: $venvPath" -ForegroundColor Green

# 3. Upgrade pip and install requirements
Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$reqFile = Join-Path $PSScriptRoot "tiny_llm_from_scratch\requirements.txt"
if (-not (Test-Path $reqFile)) {
    $reqFile = Join-Path $PSScriptRoot "requirements.txt"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pip dependency installation failed." -ForegroundColor Red
    exit 1
}

# 4. Run automated test suite
Write-Host "[INFO] Running verification test suite (35 unit & integration tests)..." -ForegroundColor Yellow
$testScript = Join-Path $PSScriptRoot "tiny_llm_from_scratch\tests\test_system.py"
& $venvPython $testScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Setup complete! All tests passed successfully." -ForegroundColor Green
    Write-Host "Run .\launch.ps1 to start the interactive chat assistant." -ForegroundColor Cyan
    Write-Host "Run .\train.ps1 to start training from scratch." -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Setup completed with test warnings. Check output above." -ForegroundColor Yellow
}
