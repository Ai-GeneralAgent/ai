# Windows PowerShell Packaging and Security Audit Script
# Verifies repository safety, 100 MiB size limits, gitignore hygiene, and test status.
# Usage: .\package.ps1

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Modern Tiny LLM - Security & Packaging Audit" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check for files > 95 MiB (GitHub hard limit is 100 MiB)
Write-Host "[1/4] Scanning repository for large files (>95 MiB)..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Path $PSScriptRoot -Recurse -File | Where-Object { 
    $_.Length -gt 95MB -and 
    $_.FullName -notmatch "\\\.git\\" -and 
    $_.FullName -notmatch "\\\.venv\\" -and 
    $_.FullName -notmatch "\\venv\\" 
}

if ($largeFiles) {
    Write-Host "[WARNING] Found files approaching or exceeding GitHub limits:" -ForegroundColor Red
    foreach ($f in $largeFiles) {
        $sizeMB = [math]::Round($f.Length / 1MB, 2)
        Write-Host "  - $($f.FullName) ($sizeMB MB)" -ForegroundColor Red
    }
} else {
    Write-Host "  [OK] No tracked source files exceeding 95 MiB found." -ForegroundColor Green
}

# 2. Check for sensitive files (API keys, .env, credentials)
Write-Host "[2/4] Scanning repository for exposed secrets or credentials..." -ForegroundColor Yellow
$secretFiles = Get-ChildItem -Path $PSScriptRoot -Recurse -File -Include "*.pem", "*.key", "id_rsa", ".env" | Where-Object { 
    $_.FullName -notmatch "\\\.git\\" -and 
    $_.FullName -notmatch "\\\.venv\\" -and 
    $_.FullName -notmatch "\\venv\\" 
}
if ($secretFiles) {
    Write-Host "[ALERT] Potential credential files detected in source tree:" -ForegroundColor Red
    foreach ($sf in $secretFiles) {
        Write-Host "  - $($sf.FullName)" -ForegroundColor Red
    }
} else {
    Write-Host "  [OK] Zero credential/private key files detected in source tree." -ForegroundColor Green
}

# 3. Check .gitignore coverage
Write-Host "[3/4] Verifying .gitignore configuration..." -ForegroundColor Yellow
$gitignorePath = Join-Path $PSScriptRoot ".gitignore"
if (Test-Path $gitignorePath) {
    $giContent = Get-Content $gitignorePath -Raw
    $requiredPatterns = @(".venv", "*.pt", "*.u16", "*.tmp", "__pycache__")
    $allPresent = $true
    foreach ($pat in $requiredPatterns) {
        if ($giContent -notmatch [regex]::Escape($pat)) {
            Write-Host "  [MISSING] .gitignore lacks rule for: $pat" -ForegroundColor Yellow
            $allPresent = $false
        }
    }
    if ($allPresent) {
        Write-Host "  [OK] .gitignore contains all required artifact exclusion patterns." -ForegroundColor Green
    }
} else {
    Write-Host "  [ERROR] .gitignore not found at repository root!" -ForegroundColor Red
}

# 4. Run automated test suite
Write-Host "[4/4] Executing verification test suite..." -ForegroundColor Yellow
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = "python" }

$testScript = Join-Path $PSScriptRoot "tiny_llm_from_scratch\tests\test_system.py"
& $venvPython $testScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "[AUDIT PASSED] Project is securely packaged and verified." -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "[AUDIT FAILED] Test suite failed. Fix regressions before packaging." -ForegroundColor Red
    exit 1
}
