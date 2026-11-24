#pragma once

#include "DlssNr_FinishedGameFrame.h"

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace DlssNr::FinishedVk
{
// One capture of the game's depth and motion (dlssnr/DlssNr_FinishedVk.h), ready to be copied into the images shared
// with the D3D12 device. The images are the capture ring's: valid until a capture four slots on writes the slot again,
// which is long after the present that takes them has recorded its copy.
struct Guides
{
    VkImage depth = VK_NULL_HANDLE;           // R32_SFLOAT, null when the capture had no depth
    VkImage motion = VK_NULL_HANDLE;          // R16G16B16A16_SFLOAT, .xy
    uint32_t depthWidth = 0, depthHeight = 0; // the game's allocation (the ring image's size)
    uint32_t motionWidth = 0, motionHeight = 0;
    FinishedGameFrame frame;
};
} // namespace DlssNr::FinishedVk
