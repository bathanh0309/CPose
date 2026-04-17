@echo off
title CPose - Research Server
echo Starting Research FastAPI Server...
set PYTHONPATH=%CD%
.venv\Scripts\activate && uvicorn research.main:app --host 0.0.0.0 --port 8000 --reload
pause
