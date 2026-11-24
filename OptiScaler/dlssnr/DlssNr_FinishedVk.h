#pragma once

// Finished Picture on a native Vulkan game: the capture half.
//
// "NR Pass at: Finished Picture" runs Neural Rendering on the picture the game presents, with the depth and motion the
// game gave its own upscaler. On D3D12 those are copied at the upscaler call and used at present (DlssNr_Late.inl). On
// a native Vulkan game this module does the first half: at the game's DLSS evaluate (EvaluateAtSeamVk, where it used to
// return) a small compute pass in the game's command buffer samples the depth and the motion vectors through the views
// NGX handed over and writes them to images of its own (a ring of four, written only in the region the upscaler reads,
// at the same texel position, as the D3D12 capture keeps the game's allocation). At present, native/VkFrameSource
// copies the newest submitted slot into images it shares with the private D3D12 device, where NR runs on the picture
// through the same code as on D3D12 (DlssNr::ApplyFinishedGameGuides): the look is the D3D12 Finished Picture's by
// construction.
//
// "Recorded but not executed": a slot is used only once its command buffer was seen in a vkQueueSubmit
// (hooks/VulkanwDx12_Hooks.cpp calls NoteSubmit); begin/reset/free of the buffer first drops it, and an unsent one
// expires after a few presents (dlssnr/DlssNr_FinishedVkRules.h).
//
// Legal on a compute-only present queue (Doom presents from async compute): the capture is a dispatch in the game's
// own command buffer, wherever its DLSS call is; the present side only copies.

#include <vulkan/vulkan.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>

#include "DlssNr_FinishedGameFrame.h"
#include "DlssNr_FinishedVkGuides.h"

#include <cstdint>
#include <string>

namespace DlssNr::FinishedVk
{
// At the game's DLSS evaluate, with Finished Picture on and Neural Rendering enabled. Records into `cmd`.
// `rayReconstruction`: a Ray Reconstruction call. Refuses (see Refusal()) when NR is to run before Super Resolution.
void Capture(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, VkPhysicalDevice physical, VkDevice device,
             bool rayReconstruction);

enum class TakeResult
{
    Taken,   // `out` is the capture; it is used up
    None,    // nothing submitted that fits (Refusal() says when it is the output size)
    NotReady // the newest capture was made on another queue than the present's and has not finished: skip this present
};

// At present `presentEpoch` (State::vulkanPresentCount, already counted) on `presentQueue`, for a picture `width` x
// `height` made on `device`: the newest capture that fits and was submitted before BeginPresent.
TakeResult Take(VkDevice device, VkQueue presentQueue, uint64_t presentEpoch, uint32_t width, uint32_t height,
                Guides& out);

// First thing in the present hook: the game's submits from now on belong to a later frame than this present.
void BeginPresent();

// Whether the last present could use a capture (not the bridge, frame generation, a colour format or space NR cannot
// take): when not, Capture records nothing and the ring is freed after the idle time. True until said otherwise.
void SetUsable(bool usable);

// The game called its DLSS within the last few presents (a capture was attempted, whether or not it worked): the game's
// own upscaler is running, so Finished Picture takes its guides from it rather than from Optical F5Low's motion
// estimate, and says what is wrong when it cannot.
bool Recent(uint64_t presentCount);

// Why the last capture could not be made (the format of the game's motion vectors, ...), empty when it could. For the
// status line while nothing can be applied.
std::string Refusal();

// Once per present (the game's present thread): drops recordings that were never submitted, retires replaced images,
// and lets go of everything after the game has stopped calling its DLSS for a while.
void OnPresent(uint64_t presentCount);

// From the vkQueueSubmit / vkQueueSubmit2 hooks, after the original call returned VK_SUCCESS: the game submitted these
// command buffers on `queue`. A relaxed load when no capture waits for its submit.
void AfterSubmit(VkQueue queue, const VkCommandBuffer* buffers, uint32_t count);
void AfterSubmit2(VkQueue queue, const VkCommandBufferSubmitInfo* buffers, uint32_t count);

// From vkBeginCommandBuffer / vkResetCommandBuffer / vkFreeCommandBuffers: what was recorded into these before is gone.
void ForgetCommandBuffers(uint32_t count, const VkCommandBuffer* buffers);

// From vkCmdExecuteCommands: a capture recorded into a secondary buffer rides on `primary`'s submit.
void NoteExecuteCommands(VkCommandBuffer primary, uint32_t count, const VkCommandBuffer* secondaries);

// The game destroys `device`: the ring, the pipelines, everything made on it. Before vkDestroyDevice.
void OnDeviceDestroyed(VkDevice device);

// The Vulkan name of a swapchain format for status lines ("B8G8R8A8_UNORM"), or "format N".
std::string FormatName(VkFormat format);
// The same for a swapchain colour space ("HDR10_HLG").
std::string SpaceName(VkColorSpaceKHR space);
} // namespace DlssNr::FinishedVk
