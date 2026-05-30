@echo off
echo ============================================================
echo Face Recognition - Windows Install
echo ============================================================

echo.
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing base packages...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [3/4] Installing prebuilt dlib (no compiler needed)...
pip install dlib-bin==19.24.6
if errorlevel 1 goto :error

echo.
echo [4/4] Installing face-recognition (without rebuilding dlib)...
pip install face-recognition==1.3.0 --no-deps
if errorlevel 1 goto :error

echo.
echo ============================================================
echo Done! Next step:
echo    python register.py
echo ============================================================
goto :eof

:error
echo.
echo ============================================================
echo Install failed. See errors above.
echo ============================================================
exit /b 1
