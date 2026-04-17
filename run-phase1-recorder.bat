@echo off
title CPose - Phase 1 Recorder
echo Starting Phase 1: RTSP Recorder...
.venv\Scripts\activate && python tools/run_phase1_recorder.py
pause
