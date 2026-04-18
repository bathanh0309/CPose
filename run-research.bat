@echo off
title CPose - Research Server
echo Starting CPose Research Benchmark Mode...
rem Usage: run-research.bat --model <path> --source <path> [--gt <ground_truth.json>]
set PYTHONPATH=%CD%
.venv\Scripts\python.exe research\benchmark_cli.py %*
pause
