# DermAI Launcher — PowerShell
Write-Host "Starting DermAI Skin Cancer Detection App..." -ForegroundColor Cyan
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$scriptDir\venv\Scripts\python.exe" -m streamlit run "$scriptDir\app.py" --server.port 8501
