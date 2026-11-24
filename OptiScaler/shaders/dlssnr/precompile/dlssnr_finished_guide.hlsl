// Finished Picture on a native Vulkan game: keeps the game's depth or motion vectors, as its DLSS call handed them over,
// in an image of OptiScaler's own (dlssnr/DlssNr_FinishedVk.cpp). Vulkan cannot copy a game's depth or motion image
// without knowing how it was made (no TRANSFER_SRC usage on most of them, depth with stencil, packed formats), but every
// image the upscaler reads can be sampled through the view it was given.
//
// Only the region the upscaler reads is written (DlssNr::ResolveGuideRegions), at the same texel position: the D3D12
// capture keeps the game's allocation and offsets the same way, so the consumer reads both alike. The values are what a
// sampled read gives (float, or the normalised value of a UNORM/SNORM/depth format), as the D3D12 consumer's typed read does.
//
// DEPTH: target R32_SFLOAT, .x of the source. Without it: target R16G16B16A16_SFLOAT, .xy of the source (a format every
// device can write as a storage image without the extended-formats feature, which the game's device may not have enabled).
//
// Built with tools in shaders/shader_tools (see build_finished_guide.bat).

[[vk::binding(0, 0)]] Texture2D<float4> gSource : register(t0);

#ifdef DEPTH
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float> gTarget : register(u0);
#else
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> gTarget : register(u0);
#endif

struct Region
{
    uint x;
    uint y;
    uint width;
    uint height;
};

[[vk::push_constant]] Region gRegion;

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gRegion.width || id.y >= gRegion.height)
        return;

    const uint2 texel = uint2(gRegion.x + id.x, gRegion.y + id.y);
    const float4 value = gSource.Load(int3(texel, 0));

#ifdef DEPTH
    gTarget[texel] = value.x;
#else
    gTarget[texel] = float4(value.xy, 0.0, 0.0);
#endif
}
