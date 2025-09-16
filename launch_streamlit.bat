@echo off
echo.
echo ========================================
echo   Diabetic Retinopathy AI System
echo   Streamlit Application Launcher
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found! Installing/updating dependencies...
pip install -r streamlit_requirements.txt

echo.
echo Launching Streamlit application...
echo The app will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py --server.port 8501

echo.
echo Application stopped.
pause 