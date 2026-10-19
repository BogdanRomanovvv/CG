@echo off
rem ---------------------------------------------------------------------------
rem Launcher for CG Lab #1 (Variant 10 - regular dodecahedron).
rem Runs the app from the project root so that shaders\*.spv are found
rem (the program loads shaders by relative path).
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

rem MinGW-w64 (UCRT64) runtime DLLs used by the build.
if exist "C:\msys64\ucrt64\bin\libstdc++-6.dll" set "PATH=C:\msys64\ucrt64\bin;%PATH%"

set "APP=build\vulkan-starter-app.exe"
if not exist "%APP%" set "APP=build-debug\vulkan-starter-app.exe"
if not exist "%APP%" set "APP=build-release\vulkan-starter-app.exe"

if not exist "%APP%" (
    echo.
    echo Executable not found. Build the project first, for example:
    echo     cmake --preset mingw-debug
    echo     cmake --build build-debug --parallel
    echo.
    pause
    exit /b 1
)

echo Starting %APP% ...
"%APP%"
endlocal
