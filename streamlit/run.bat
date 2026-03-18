@echo off
echo Starting DermAI Skin Cancer Detection App...
echo.
call "%~dp0venv\Scripts\activate.bat"
echo Launching Streamlit on http://localhost:8501
echo Press Ctrl+C to stop.
echo.
"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.port 8501
pause
