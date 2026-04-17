@echo off
title CPose - Phase 2 Analyzer
echo Starting Phase 2: Offline Label Analyzer...
.venv\Scripts\activate && python tools/run_phase2_analyzer.py
pause
