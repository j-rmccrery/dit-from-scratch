@echo off
rem Build dit-rs with the Candle CUDA backend on Windows.
rem Needs: Visual Studio Build Tools (C++ workload), CUDA toolkit (nvcc), rustup MSVC toolchain.
rem   rustup toolchain install stable-x86_64-pc-windows-msvc
rem Output: target-cuda\release\dit-rs.exe   Extra args are passed to cargo.
setlocal

rem nvcc uses cl.exe as its host compiler, so load the MSVC developer environment
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere.exe not found: install Visual Studio Build Tools & exit /b 1)
rem `call` stops for /f from stripping the quotes around a path with spaces
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo MSVC C++ tools not found: add the "Desktop development with C++" workload & exit /b 1)
rem vcvars64.bat calls vswhere.exe by name, so put the Installer folder on PATH first
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem CUDA_PATH is set by the CUDA installer, but shells opened before the install won't see it
if not defined CUDA_PATH (
    for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%d"
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (echo nvcc not found: install the CUDA toolkit & exit /b 1)
echo using CUDA at %CUDA_PATH%
set "PATH=%CUDA_PATH%\bin;%USERPROFILE%\.cargo\bin;%PATH%"

rem GPU architecture for the kernels, e.g. 89 for RTX 40-series; read from the driver if not given
if not defined CUDA_COMPUTE_CAP (
    for /f "usebackq delims=" %%c in (`nvidia-smi --query-gpu=compute_cap --format=csv^,noheader`) do if not defined CUDA_COMPUTE_CAP set "CUDA_COMPUTE_CAP=%%c"
)
set "CUDA_COMPUTE_CAP=%CUDA_COMPUTE_CAP:.=%"
echo building for compute capability %CUDA_COMPUTE_CAP%

cd /d "%~dp0"
set CARGO_TARGET_DIR=target-cuda
cargo +stable-x86_64-pc-windows-msvc build --release --features cuda %*
