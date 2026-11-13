@echo off
title Claude SkyrimNet Proxy
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" proxy.py
) else (
  py -3 proxy.py
)
pause
