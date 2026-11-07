# Run the test suite on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\test.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\test.ps1 -k tiling
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".venv\Scripts\Activate.ps1") { . .venv\Scripts\Activate.ps1 }
python -m pytest -q @args
exit $LASTEXITCODE
