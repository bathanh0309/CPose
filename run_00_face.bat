@echo off
REM CPose - Module 0: Face Recognition + Anti-Spoofing
REM Usage: run_00_face.bat [source] [max_frames]
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "SOURCE=%~1"
if "%SOURCE%"=="" set "SOURCE=0"
set "MAX_FRAMES=%~2"
if "%MAX_FRAMES%"=="" set "MAX_FRAMES=0"
set "OUTPUT=dataset\outputs\0_face"
set "CONFIG=configs\profiles\dev.yaml"
set "GALLERY=data\face"
set "LIVENESS=models\face_antispoof\best_model_quantized.onnx"
set "RUN_EVERY=1"
echo [CPose] Face Recognition - Source: %SOURCE%, Max frames: %MAX_FRAMES%
"%PYTHON%" -m src.modules.face_recognizer.main ^
  --source "%SOURCE%" ^
  --max-frames %MAX_FRAMES% ^
  --output "%OUTPUT%" ^
  --config "%CONFIG%" ^
  --gallery "%GALLERY%" ^
  --liveness "%LIVENESS%" ^
  --run-every %RUN_EVERY%
popd
echo  Source     : %SOURCE%
echo  Output     : %OUTPUT%
echo  Gallery    : %GALLERY%
echo  Config     : %CONFIG%
echo  Liveness   : %LIVENESS%
echo  Stop       : press Q or Esc in the preview window
echo ============================================================

"%PYTHON%" -m src.modules.face.main ^
  --source "%SOURCE%" ^
  --output "%OUTPUT%" ^
  --config "%CONFIG%" ^
  --gallery "%GALLERY%" ^
  --liveness-model "%LIVENESS%" ^
  --run-every-n-frames %RUN_EVERY% ^
  --max-frames %MAX_FRAMES% ^
  --anti-spoof ^
  --preview ^
  --save-video

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Face module failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Face module complete. Results in: %OUTPUT%
popd
endlocal
