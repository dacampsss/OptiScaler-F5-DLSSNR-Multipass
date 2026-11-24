// host_low_raw for Vulkan: VkAux.hlsli, then RawLow.hlsl (which includes the frozen v59 shader).
#include "VkAux.hlsli"
// The storage targets of the builds that write one known format (the compile script defines which): declared with that
// format, as dlssnr_spatial.hlsl does, instead of the one dxc would infer from the element type (a float4 is Rgba32f). Only
// where every RWTexture2D of the build is that format.
#if defined(VK_TARGET_RGBA16F)
#define RWTexture2D [[vk::image_format("rgba16f")]] RWTexture2D
#elif defined(VK_TARGET_R32F)
#define RWTexture2D [[vk::image_format("r32f")]] RWTexture2D
#endif
#include "RawLow.hlsl"
