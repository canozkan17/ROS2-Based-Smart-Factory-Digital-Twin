@echo off
REM StartSystem.bat - run the project start script inside WSL
pushd "%~dp0"

wsl -- bash -lc "cd /home/canozkan/Capstone_Project && ./start_system.sh"

popd
pause
