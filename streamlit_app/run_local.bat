@echo off
echo ========================================
echo AI DAO Hedge Fund - Streamlit App
echo ========================================
echo.
echo Starting Streamlit app...
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run app.py

pause
