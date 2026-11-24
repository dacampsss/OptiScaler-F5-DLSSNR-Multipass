@echo off
setlocal
rem Rebuild the two guide-capture variants (Finished Picture on Vulkan) with the bundled DXC and header generator.
set "DXC=%~dp0..\..\shader_tools\dxc.exe"
set "HEADER=%~dp0..\..\shader_tools\create_header.py"
set "SOURCE=%~dp0dlssnr_finished_guide.hlsl"

"%DXC%" -spirv -T cs_6_0 -E CSMain -O3 -Qstrip_debug -D DEPTH -Cc -Vi "%SOURCE%" -Fo "%~dp0dlssnr_finished_guide_depth_Shader_Vk.spv"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_finished_guide_depth_Shader_Vk.spv" "%~dp0dlssnr_finished_guide_depth_Shader_Vk.h" dlssnr_finished_guide_depth_spv
if errorlevel 1 exit /b 1

"%DXC%" -spirv -T cs_6_0 -E CSMain -O3 -Qstrip_debug -Cc -Vi "%SOURCE%" -Fo "%~dp0dlssnr_finished_guide_motion_Shader_Vk.spv"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_finished_guide_motion_Shader_Vk.spv" "%~dp0dlssnr_finished_guide_motion_Shader_Vk.h" dlssnr_finished_guide_motion_spv
if errorlevel 1 exit /b 1

echo Finished Picture guide shaders rebuilt.
