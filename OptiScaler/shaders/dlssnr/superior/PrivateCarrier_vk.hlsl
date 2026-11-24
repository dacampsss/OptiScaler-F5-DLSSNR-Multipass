// The private DLSS carrier shaders (PrivateCarrier.hlsl) for Vulkan: VkAux.hlsli declares the two auxiliary textures the
// frozen v59 shader names, and the one PrivateCarrier.hlsl renames (gAux2 -> gUnusedPrivateAux) is declared here under its
// new name, before the file is included. None of them is read by a pass built from this include except gAux (t4), which
// VkAux.hlsli puts on binding 9. Compiled by tools/compile_superior_vk.ps1.
#include "VkAux.hlsli"
[[vk::binding(10, 0)]]
Texture2D<float4> gUnusedPrivateAux : register(t5);
// The frozen file declares gTarget / gKeep without a format; PrivateCarrier.hlsl declares them again, with one.
#define gTarget gFrozenTarget
#define gKeep gFrozenKeep
#define VK_PRIVATE_TARGETS 1
#include "PrivateCarrier.hlsl"
