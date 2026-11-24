#pragma once

#include <cstdint>

namespace DlssNr
{
// What the game's own upscaler call said about one frame, kept at that call for Finished Picture on a Vulkan game
// (dlssnr/DlssNr_FinishedVk.h): the same values the D3D12 capture (DlssNr_Late.inl, Late::Capture) keeps in its slot's
// DlssNrFrameInfo. Plain data, so the Vulkan side, the frame contract (native/FrameContract.h) and the D3D12 consumer
// share one struct without sharing any API.
struct FinishedGameFrame
{
    bool reset = false;         // the game asked its upscaler to forget its history (carried over dropped captures)
    bool depthInverted = false; // DLSS create flag DepthInverted
    bool mvLowRes = false;      // DLSS create flag MVLowRes
    bool rayReconstruction = false;
    bool hasDepth = false;                      // a depth the capture could read (NR runs without one)
    uint32_t outputWidth = 0, outputHeight = 0; // the upscaler's output: must be the swapchain's size
    uint32_t renderWidth = 0, renderHeight = 0; // DLSS render subrect (0: the whole allocation)
    uint32_t depthBaseX = 0, depthBaseY = 0;
    uint32_t motionBaseX = 0, motionBaseY = 0;
    float mvScaleX = 1.0f, mvScaleY = 1.0f;
    uint64_t captureEpoch = 0; // State::vulkanPresentCount when the game's command buffer was recorded
};
} // namespace DlssNr
