@echo off
echo ⚾ 少年野球フォーム分析ツール を起動します...
echo.
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app.py --server.port 8501
pause
