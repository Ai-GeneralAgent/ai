@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   TinyGPT - Remove Duplicate Text
echo ==========================================
echo.

if not exist "merge_previous_dataset.py" (
    echo ERROR: merge_previous_dataset.py not found.
    pause
    exit /b 1
)

if not exist "data_merged_updated.txt" (
    echo ERROR: data_merged_updated.txt not found.
    pause
    exit /b 1
)

if exist "data_deduplicated.txt" (
    echo ERROR: data_deduplicated.txt already exists.
    echo Rename or back it up before running again.
    pause
    exit /b 1
)

python merge_previous_dataset.py --files data_merged_updated.txt --output data_deduplicated.txt

if errorlevel 1 (
    echo.
    echo ERROR: Deduplication failed.
    pause
    exit /b 1
)

echo.
echo Finished. Review data_deduplicated.txt before training.
echo Your original database was not overwritten.
pause
