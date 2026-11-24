#include "pch.h"

#include "DlssNr_FinishedVk.h"
#include "DlssNr_FinishedVkRules.h"

#include <Config.h>
#include <State.h>
#include <hooks/Vulkan_Hooks.h>
#include <native/VkFrameSource.h>
#include <shaders/dlssnr/DlssNr_Guides.h>
#include <shaders/dlssnr/precompile/dlssnr_finished_guide_depth_Shader_Vk.h>
#include <shaders/dlssnr/precompile/dlssnr_finished_guide_motion_Shader_Vk.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace DlssNr::FinishedVk
{
namespace
{
using namespace FinishedVkRules;

constexpr VkFormat kDepthFormat = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

// Presents after which a replaced ring image is destroyed: the frames in flight that read it are long done.
constexpr uint64_t kRetirePresents = 8;
// The game has stopped calling its DLSS: everything goes after this many presents and this much time (the usual NR
// off delay on Vulkan is ten seconds, DlssNrFeature_Vk.cpp kOffReleaseVk).
constexpr uint64_t kIdlePresents = 120;
constexpr std::chrono::seconds kIdleTime { 10 };
// A capture this many presents old or less counts as "the game calls its DLSS".
constexpr uint64_t kRecentPresents = 8;

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Retired
{
    Image image;
    uint64_t after = 0; // the present count from which it is destroyed
};

struct SlotImages
{
    Image depth, motion;
    VkImageView depthOnly =
        VK_NULL_HANDLE; // the depth-aspect view of the game's depth+stencil image, until the slot is reused
};

// Everything made on the game's device. One device at a time (the one its DLSS runs on).
struct Objects
{
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    bool failed = false;
    VkPhysicalDeviceMemoryProperties memory {};
    std::vector<uint32_t> families;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline depthPipeline = VK_NULL_HANDLE;
    VkPipeline motionPipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kSlots][2] = {}; // per slot: [0] depth, [1] motion

    SlotImages slots[kSlots];
    uint32_t depthWidth = 0, depthHeight = 0, motionWidth = 0, motionHeight = 0; // the ring's sizes
    std::vector<Retired> retired;
    uint64_t captures = 0;

    // One timeline semaphore per queue the game submits captures on, signalled by an empty submit right behind the
    // game's (submit order puts everything the game queued before it). A present on ANOTHER queue reads it to tell the
    // capture has finished, and slot reuse reads it to tell the GPU is done with the slot. Only on a device that has
    // the timeline feature on (VulkanHooks::TimelineSemaphoresOn); without, other-queue captures are not used and reuse
    // goes by age.
    struct QueueTimeline
    {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t value = 0; // the last value an empty submit signalled
    };
    bool timelineOn = false;
    std::unordered_map<uint64_t, QueueTimeline> timelines; // by VkQueue

    bool Built() const { return pipelineLayout != VK_NULL_HANDLE && motionPipeline != VK_NULL_HANDLE; }
};

std::mutex g_mutex;
GuideRing g_ring;
Objects g_obj;
std::atomic<uint32_t> g_armed { 0 };                // g_ring.Armed(), for the submit hooks' fast path
std::atomic<uint64_t> g_lastCapture { UINT64_MAX }; // the present count at the last successful capture
std::atomic<uint64_t> g_lastAttempt { UINT64_MAX }; // the present count at the last Capture call, success or refusal
std::atomic<bool> g_usable { true };                // the last present could use a capture (SetUsable)
std::atomic<uint64_t> g_submitClock { 0 };          // numbers the game's submits that carried a capture
// g_submitClock when the present hook was entered. One global value, set by the present thread: a game that presents
// from another thread than it submits on, or two swapchains presenting at once, can see a snapshot from the other
// present. The worst case is one frame's guides one frame early or late (the same size and window rules still apply).
std::atomic<uint64_t> g_snapshot { UINT64_MAX };
std::chrono::steady_clock::time_point g_lastCaptureTime {};
std::string g_refusal; // under g_mutex

uint64_t PresentNow() { return State::Instance().vulkanPresentCount.load(std::memory_order_relaxed); }

uint64_t Handle(VkCommandBuffer cmd) { return (uint64_t) (uintptr_t) cmd; }

void PublishArmed() { g_armed.store(g_ring.Armed(), std::memory_order_relaxed); }

uint32_t FindMemory(const VkPhysicalDeviceMemoryProperties& memory, uint32_t typeBits)
{
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) != 0 &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            return i;
    }

    return UINT32_MAX;
}

void DestroyImage(VkDevice device, Image& image)
{
    if (image.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, image.view, nullptr);

    if (image.image != VK_NULL_HANDLE)
        vkDestroyImage(device, image.image, nullptr);

    if (image.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, image.memory, nullptr);

    image = Image {};
}

// A storage image the copy at present can also read. Concurrent across the device's queue families: the game's DLSS
// call and its present can be on different queues (Doom presents from async compute), and the image never changes
// layout after its first use, so there is no ownership to hand over.
bool MakeImage(Objects& o, Image& out, uint32_t width, uint32_t height, VkFormat format)
{
    VkImageCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = { width, height, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (o.families.size() > 1)
    {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = (uint32_t) o.families.size();
        info.pQueueFamilyIndices = o.families.data();
    }
    else
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(o.device, &info, nullptr, &out.image) != VK_SUCCESS)
        return false;

    VkMemoryRequirements need {};
    vkGetImageMemoryRequirements(o.device, out.image, &need);

    VkMemoryAllocateInfo allocate {};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = need.size;
    allocate.memoryTypeIndex = FindMemory(o.memory, need.memoryTypeBits);

    if (allocate.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(o.device, &allocate, nullptr, &out.memory) != VK_SUCCESS ||
        vkBindImageMemory(o.device, out.image, out.memory, 0) != VK_SUCCESS)
    {
        DestroyImage(o.device, out);
        return false;
    }

    VkImageViewCreateInfo view {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = out.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    if (vkCreateImageView(o.device, &view, nullptr, &out.view) != VK_SUCCESS)
    {
        DestroyImage(o.device, out);
        return false;
    }

    return true;
}

void DestroyRetired(Objects& o, bool all, uint64_t presentNow = 0)
{
    std::erase_if(o.retired,
                  [&](Retired& old)
                  {
                      if (!all && presentNow < old.after)
                          return false;

                      DestroyImage(o.device, old.image);
                      return true;
                  });
}

// Everything on the device. The caller knows no work of ours is in flight (the game destroys the device, or no capture
// was made for many frames).
void Destroy(Objects& o)
{
    if (o.device == VK_NULL_HANDLE)
    {
        o = Objects {};
        return;
    }

    for (auto& slot : o.slots)
    {
        DestroyImage(o.device, slot.depth);
        DestroyImage(o.device, slot.motion);

        if (slot.depthOnly != VK_NULL_HANDLE)
            vkDestroyImageView(o.device, slot.depthOnly, nullptr);

        slot.depthOnly = VK_NULL_HANDLE;
    }

    DestroyRetired(o, true);

    if (o.depthPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(o.device, o.depthPipeline, nullptr);

    if (o.motionPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(o.device, o.motionPipeline, nullptr);

    for (auto& [queue, timeline] : o.timelines)
    {
        if (timeline.semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(o.device, timeline.semaphore, nullptr);
    }

    if (o.pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(o.device, o.pool, nullptr);

    if (o.pipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(o.device, o.pipelineLayout, nullptr);

    if (o.setLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(o.device, o.setLayout, nullptr);

    o = Objects {};
}

bool BuildPipeline(Objects& o, const unsigned char* code, size_t size, VkPipeline& pipeline)
{
    VkShaderModuleCreateInfo moduleInfo {};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = size;
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code);

    VkShaderModule module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(o.device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "CSMain";
    info.layout = o.pipelineLayout;

    const VkResult made = vkCreateComputePipelines(o.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(o.device, module, nullptr);
    return made == VK_SUCCESS;
}

// The two pipelines and their descriptor sets, on first use (and again after the game's device changes). A failure is
// kept
// (`failed`): the capture does not try again on this device, and the status line says why.
bool EnsureObjects(VkPhysicalDevice physical, VkDevice device)
{
    if (g_obj.device == device && g_obj.Built())
        return true;

    if (g_obj.device == device && g_obj.failed)
        return false;

    // Another device than the one the objects were made on: the old one is gone with its device or about to be (the
    // game made a new one), so what was made there is forgotten rather than destroyed on a dead handle.
    if (g_obj.device != VK_NULL_HANDLE && g_obj.device != device)
        g_obj = Objects {};

    g_obj.device = device;
    g_obj.physical = physical;
    g_obj.families = native::VkFrameSource::FamiliesOf(device);
    vkGetPhysicalDeviceMemoryProperties(physical, &g_obj.memory);

    const auto fail = [&](const char* what)
    {
        LOG_WARN("DLSS-NR Vulkan finished picture: the capture pass's {} could not be made", what);
        auto keep = std::move(g_obj.families);
        Destroy(g_obj);
        g_obj.device = device;
        g_obj.physical = physical;
        g_obj.failed = true;
        g_obj.families = std::move(keep);
        g_refusal = std::string("Finished Picture on Vulkan could not set up its capture pass (") + what + ").";
        return false;
    };

    // The images are shared across the device's queue families (the game's DLSS and its present can be on different
    // queues). Not knowing the families is refused rather than guessed: an exclusive image used from two families is a
    // validation error and garbage on some drivers.
    if (g_obj.families.empty())
        return fail("queue families (the device was made before OptiScaler could see it)");

    // The format the motion goes to must be writable as a storage image and readable by a transfer; R32_SFLOAT is
    // mandatory for both.
    VkFormatProperties properties {};
    vkGetPhysicalDeviceFormatProperties(physical, kMotionFormat, &properties);
    constexpr VkFormatFeatureFlags kNeed = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;

    if ((properties.optimalTilingFeatures & kNeed) != kNeed)
        return fail("storage format");

    // Binding 0 the game's view (sampled, read through the texel fetch), binding 1 the ring image (storage)
    VkDescriptorSetLayoutBinding bindings[2] {};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &g_obj.setLayout) != VK_SUCCESS)
        return fail("descriptor set layout");

    const VkPushConstantRange push { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t) };
    VkPipelineLayoutCreateInfo layoutCreate {};
    layoutCreate.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreate.setLayoutCount = 1;
    layoutCreate.pSetLayouts = &g_obj.setLayout;
    layoutCreate.pushConstantRangeCount = 1;
    layoutCreate.pPushConstantRanges = &push;

    if (vkCreatePipelineLayout(device, &layoutCreate, nullptr, &g_obj.pipelineLayout) != VK_SUCCESS)
        return fail("pipeline layout");

    constexpr uint32_t kSets = kSlots * 2;
    const VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSets },
                                           { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSets } };
    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = kSets;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &g_obj.pool) != VK_SUCCESS)
        return fail("descriptor pool");

    VkDescriptorSetLayout layouts[kSets];
    std::fill(std::begin(layouts), std::end(layouts), g_obj.setLayout);
    VkDescriptorSet sets[kSets] = {};
    VkDescriptorSetAllocateInfo allocate {};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = g_obj.pool;
    allocate.descriptorSetCount = kSets;
    allocate.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(device, &allocate, sets) != VK_SUCCESS)
        return fail("descriptor sets");

    for (uint32_t i = 0; i < kSlots; ++i)
    {
        g_obj.sets[i][0] = sets[2 * i];
        g_obj.sets[i][1] = sets[2 * i + 1];
    }

    if (!BuildPipeline(g_obj, dlssnr_finished_guide_depth_spv, sizeof(dlssnr_finished_guide_depth_spv),
                       g_obj.depthPipeline) ||
        !BuildPipeline(g_obj, dlssnr_finished_guide_motion_spv, sizeof(dlssnr_finished_guide_motion_spv),
                       g_obj.motionPipeline))
        return fail("compute pipelines");

    // Timeline semaphores only where the device was made with the feature (a semaphore of that type is an error
    // otherwise)
    g_obj.timelineOn = VulkanHooks::TimelineSemaphoresOn(device);

    if (!g_obj.timelineOn)
        LOG_WARN(
            "DLSS-NR Vulkan finished picture: the game's device has no timeline semaphores; captures made on another "
            "queue than the present's are not used, and slots are reused by age alone");

    LOG_INFO("DLSS-NR Vulkan finished picture: capture pass ready on the game's device ({} queue families)",
             g_obj.families.size());
    return true;
}

// The ring images for guides of this size: made on first use, replaced (the old ones retired, not destroyed under the
// frames that may still read them) when the game's allocation changes.
bool EnsureRing(Objects& o, bool depth, uint32_t width, uint32_t height)
{
    uint32_t& ringWidth = depth ? o.depthWidth : o.motionWidth;
    uint32_t& ringHeight = depth ? o.depthHeight : o.motionHeight;

    if (ringWidth == width && ringHeight == height)
        return true;

    for (auto& slot : o.slots)
    {
        Image& image = depth ? slot.depth : slot.motion;

        if (image.image != VK_NULL_HANDLE)
            o.retired.push_back({ image, PresentNow() + kRetirePresents });

        image = Image {};
    }

    // Slots of the old size hold nothing a present may still take
    g_ring.Clear();
    ringWidth = ringHeight = 0;

    for (auto& slot : o.slots)
    {
        if (!MakeImage(o, depth ? slot.depth : slot.motion, width, height, depth ? kDepthFormat : kMotionFormat))
        {
            for (auto& made : o.slots)
            {
                Image& image = depth ? made.depth : made.motion;

                if (image.image != VK_NULL_HANDLE)
                    DestroyImage(o.device, image);
            }

            return false;
        }
    }

    ringWidth = width;
    ringHeight = height;
    LOG_INFO("DLSS-NR Vulkan finished picture: the {} ring is {}x{}", depth ? "depth" : "motion", width, height);
    return true;
}

// The region of the game's guide into the slot's ring image, in the game's command buffer.
void RecordPass(const Objects& o, VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet set, VkImageView source,
                const Image& target, const GuideRegion& region)
{
    const VkDescriptorImageInfo sourceInfo { VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    const VkDescriptorImageInfo targetInfo { VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_GENERAL };
    const VkWriteDescriptorSet writes[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sourceInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &targetInfo,
          nullptr, nullptr },
    };
    vkUpdateDescriptorSets(o.device, 2, writes, 0, nullptr);

    // The previous contents are not needed (only the region is written, and only the region is read): UNDEFINED as the
    // old layout lets the driver drop them. The image stays in GENERAL from here on, which the copy at present reads
    // too.
    VkImageMemoryBarrier toWrite {};
    toWrite.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toWrite.srcAccessMask = 0;
    toWrite.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toWrite.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toWrite.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toWrite.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toWrite.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toWrite.image = target.image;
    toWrite.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toWrite);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, o.pipelineLayout, 0, 1, &set, 0, nullptr);
    const uint32_t constants[4] = { region.x, region.y, region.width, region.height };
    vkCmdPushConstants(cmd, o.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
    // [numthreads(8, 8, 1)]
    vkCmdDispatch(cmd, (region.width + 7) / 8, (region.height + 7) / 8, 1);

    // Visible to the copy at present, which follows on the present's queue (the barrier's second scope reaches later
    // commands on this queue; the game's own semaphores order the other queues).
    VkImageMemoryBarrier toRead = toWrite;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toRead);
}

bool ImageViewOk(const NVSDK_NGX_Resource_VK* resource)
{
    return resource != nullptr && resource->Type == NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW &&
           resource->Resource.ImageViewInfo.ImageView != VK_NULL_HANDLE &&
           resource->Resource.ImageViewInfo.Image != VK_NULL_HANDLE;
}
} // namespace

std::string FormatName(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "B8G8R8A8_SRGB";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "R8G8B8A8_SRGB";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return "A2B10G10R10_UNORM_PACK32";
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return "A2R10G10B10_UNORM_PACK32";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "R16G16B16A16_SFLOAT";
    case VK_FORMAT_R16G16B16A16_UNORM:
        return "R16G16B16A16_UNORM";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return "B10G11R11_UFLOAT_PACK32";
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
        return "R5G6B5_UNORM_PACK16";
    case VK_FORMAT_B5G6R5_UNORM_PACK16:
        return "B5G6R5_UNORM_PACK16";
    default:
        return "format " + std::to_string((int) format);
    }
}

namespace
{
// The queue's timeline, made on first use. Under g_mutex. Null when the device has none or it could not be made.
Objects::QueueTimeline* TimelineFor(uint64_t queue)
{
    if (!g_obj.timelineOn || g_obj.device == VK_NULL_HANDLE)
        return nullptr;

    auto& timeline = g_obj.timelines[queue];

    if (timeline.semaphore == VK_NULL_HANDLE)
    {
        VkSemaphoreTypeCreateInfo type {};
        type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        info.pNext = &type;

        if (vkCreateSemaphore(g_obj.device, &info, nullptr, &timeline.semaphore) != VK_SUCCESS)
        {
            timeline.semaphore = VK_NULL_HANDLE;
            g_obj.timelineOn = false; // not tried again; same-queue captures still work
            LOG_WARN("DLSS-NR Vulkan finished picture: a timeline semaphore could not be made");
            return nullptr;
        }
    }

    return &timeline;
}

// Whether the GPU has reached the signal that followed the slot's submit (polled, never waited for). True for a slot
// with no value, and for one whose queue has no timeline (nothing to ask). Under g_mutex.
bool SlotGpuDone(const GuideRing::Slot& slot)
{
    if (slot.value == 0)
        return true;

    const auto it = g_obj.timelines.find(slot.queue);

    if (it == g_obj.timelines.end() || it->second.semaphore == VK_NULL_HANDLE)
        return true;

    return VulkanHooks::WaitTimelineSemaphore(g_obj.device, it->second.semaphore, slot.value, 0);
}
} // namespace

std::string SpaceName(VkColorSpaceKHR space)
{
    switch (space)
    {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        return "SRGB_NONLINEAR";
    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
        return "DISPLAY_P3_NONLINEAR";
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        return "EXTENDED_SRGB_LINEAR";
    case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
        return "DISPLAY_P3_LINEAR";
    case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
        return "DCI_P3_NONLINEAR";
    case VK_COLOR_SPACE_BT709_LINEAR_EXT:
        return "BT709_LINEAR";
    case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
        return "BT709_NONLINEAR";
    case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
        return "BT2020_LINEAR";
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
        return "HDR10_ST2084";
    case VK_COLOR_SPACE_DOLBYVISION_EXT:
        return "DOLBYVISION";
    case VK_COLOR_SPACE_HDR10_HLG_EXT:
        return "HDR10_HLG";
    case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
        return "ADOBERGB_LINEAR";
    case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
        return "ADOBERGB_NONLINEAR";
    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
        return "PASS_THROUGH";
    case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        return "EXTENDED_SRGB_NONLINEAR";
    default:
        return "colour space " + std::to_string((int) space);
    }
}

void Capture(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, VkPhysicalDevice physical, VkDevice device,
             bool rayReconstruction)
{
    if (cmd == VK_NULL_HANDLE || params == nullptr || physical == VK_NULL_HANDLE || device == VK_NULL_HANDLE)
        return;

    // Every call counts, refused or not: the present side shows what is wrong only while the game is seen to call its
    // DLSS
    g_lastAttempt.store(PresentNow(), std::memory_order_relaxed);
    const auto& cfg = *Config::Instance();

    // The model before Super Resolution, and "generate before SR, apply after" are the D3D12 residual placements (mode
    // 2 of DlssNr_Dx12.cpp), which are not built on Vulkan; the D3D12 path refuses Ray Reconstruction with them too.
    if (cfg.DlssNrRunBeforeSr.value_or_default() || cfg.DlssNrDeferredDlss.value_or_default())
    {
        std::lock_guard lock(g_mutex);
        g_refusal = "Finished Picture on Vulkan runs on the whole finished picture only: switch off \"Run the model "
                    "before Super Resolution\" and \"Generate before SR\".";
        return;
    }

    // The last present could not use a capture (the bridge, frame generation, a colour format or space NR cannot take):
    // nothing is recorded into the game's command buffer, and the ring is freed after the idle time
    if (!g_usable.load(std::memory_order_relaxed))
        return;

    NVSDK_NGX_Resource_VK* depth = nullptr;
    NVSDK_NGX_Resource_VK* motion = nullptr;
    NVSDK_NGX_Resource_VK* output = nullptr;
    params->Get(NVSDK_NGX_Parameter_Depth, (void**) &depth);
    params->Get(NVSDK_NGX_Parameter_MotionVectors, (void**) &motion);
    params->Get(NVSDK_NGX_Parameter_Output, (void**) &output);

    // Motion is required, depth is not (NR runs without it), as on D3D12
    if (!ImageViewOk(motion) || !ImageViewOk(output))
    {
        std::lock_guard lock(g_mutex);
        g_refusal = "Waiting for the game's movement data.";
        return;
    }

    const auto& motionInfo = motion->Resource.ImageViewInfo;
    const GuideRead motionRead = ReadMotion(motionInfo.Format);

    if (motionRead == GuideRead::Refused)
    {
        std::lock_guard lock(g_mutex);
        g_refusal =
            "Finished Picture on Vulkan cannot read the game's motion vectors (" + FormatName(motionInfo.Format) + ").";
        static VkFormat saidMotion = VK_FORMAT_UNDEFINED;

        if (saidMotion != motionInfo.Format)
        {
            saidMotion = motionInfo.Format;
            LOG_WARN("DLSS-NR Vulkan finished picture: {}", g_refusal);
        }

        return;
    }

    GuideRead depthRead = GuideRead::Refused;
    const VkImageAspectFlags depthAspect =
        ImageViewOk(depth) ? depth->Resource.ImageViewInfo.SubresourceRange.aspectMask : 0;

    if (ImageViewOk(depth))
    {
        depthRead = ReadDepth(depth->Resource.ImageViewInfo.Format, depthAspect);
    }

    // What the game's call said about this frame: the same values the D3D12 capture keeps (Late::Capture)
    FinishedGameFrame frame;
    unsigned int flags = 0, gameReset = 0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &flags);
    params->Get(NVSDK_NGX_Parameter_Reset, &gameReset);
    frame.reset = gameReset != 0;
    frame.depthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    frame.mvLowRes = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;
    frame.rayReconstruction = rayReconstruction;
    frame.hasDepth = depthRead != GuideRead::Refused;
    frame.outputWidth = output->Resource.ImageViewInfo.Width;
    frame.outputHeight = output->Resource.ImageViewInfo.Height;
    uint32_t declaredWidth = 0, declaredHeight = 0;
    params->Get(NVSDK_NGX_Parameter_OutWidth, &declaredWidth);
    params->Get(NVSDK_NGX_Parameter_OutHeight, &declaredHeight);

    if (declaredWidth != 0 && declaredHeight != 0)
    {
        frame.outputWidth = declaredWidth;
        frame.outputHeight = declaredHeight;
    }

    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &frame.renderWidth);
    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, &frame.renderHeight);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, &frame.depthBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, &frame.depthBaseY);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, &frame.motionBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, &frame.motionBaseY);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &frame.mvScaleX);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &frame.mvScaleY);
    frame.captureEpoch = PresentNow();

    if (frame.outputWidth == 0 || frame.outputHeight == 0)
    {
        std::lock_guard lock(g_mutex);
        g_refusal = "Waiting for the game's movement data.";
        return;
    }

    // Exactly the regions the consumer reads (the same function the D3D12 capture and the Vulkan model use)
    const uint32_t depthAllocWidth = frame.hasDepth ? depth->Resource.ImageViewInfo.Width : 0;
    const uint32_t depthAllocHeight = frame.hasDepth ? depth->Resource.ImageViewInfo.Height : 0;
    const auto regions =
        ResolveGuideRegions({ depthAllocWidth, depthAllocHeight }, { motionInfo.Width, motionInfo.Height },
                            { frame.renderWidth, frame.renderHeight }, { frame.outputWidth, frame.outputHeight },
                            frame.mvLowRes, frame.depthBaseX, frame.depthBaseY, frame.motionBaseX, frame.motionBaseY);

    if (!regions.motion.valid())
    {
        std::lock_guard lock(g_mutex);
        g_refusal = "Waiting for the game's movement data.";
        return;
    }

    if (frame.hasDepth && !regions.depth.valid())
        frame.hasDepth = false;

    // The depth-aspect view of a depth+stencil image (a view over both aspects cannot be sampled); kept in the slot
    // until it is written again, four captures on. Made before the slot is known; a depth that cannot be viewed is no
    // depth.
    VkImageView depthOnlyView = VK_NULL_HANDLE;
    VkImageView depthSource = VK_NULL_HANDLE;

    if (frame.hasDepth)
    {
        const auto& depthInfo = depth->Resource.ImageViewInfo;
        depthSource = depthInfo.ImageView;

        if (depthRead == GuideRead::DepthOnly)
        {
            VkImageViewCreateInfo view {};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = depthInfo.Image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = depthInfo.Format;
            view.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, depthInfo.SubresourceRange.baseMipLevel, 1,
                                      depthInfo.SubresourceRange.baseArrayLayer, 1 };

            if (vkCreateImageView(device, &view, nullptr, &depthOnlyView) == VK_SUCCESS)
                depthSource = depthOnlyView;
            else
            {
                depthOnlyView = VK_NULL_HANDLE;
                depthSource = VK_NULL_HANDLE;
                frame.hasDepth = false;
            }
        }
    }

    std::lock_guard lock(g_mutex);

    // Said once per format, under the lock: the game's render threads can call this at the same time
    static VkFormat saidDepth = VK_FORMAT_UNDEFINED;

    if (depthRead == GuideRead::Refused && ImageViewOk(depth) && saidDepth != depth->Resource.ImageViewInfo.Format)
    {
        saidDepth = depth->Resource.ImageViewInfo.Format;
        LOG_WARN("DLSS-NR Vulkan finished picture: the game's depth ({}) cannot be read; NR runs without depth",
                 FormatName(depth->Resource.ImageViewInfo.Format));
    }

    if (!EnsureObjects(physical, device))
    {
        if (depthOnlyView != VK_NULL_HANDLE)
            vkDestroyImageView(device, depthOnlyView, nullptr);

        return; // g_refusal says why
    }

    ++g_obj.captures;
    DestroyRetired(g_obj, false, frame.captureEpoch);

    if (!EnsureRing(g_obj, false, motionInfo.Width, motionInfo.Height) ||
        (frame.hasDepth && !EnsureRing(g_obj, true, depthAllocWidth, depthAllocHeight)))
    {
        if (depthOnlyView != VK_NULL_HANDLE)
            vkDestroyImageView(device, depthOnlyView, nullptr);

        g_refusal = "Finished Picture on Vulkan could not allocate its capture images.";
        return;
    }

    size_t index = 0;

    // No slot free (a game that captures several times per frame, or frames stuck in flight): this frame has no guides,
    // and the present side says it is waiting
    if (!g_ring.Begin(Handle(cmd), frame.captureEpoch, frame, index, SlotGpuDone))
    {
        if (depthOnlyView != VK_NULL_HANDLE)
            vkDestroyImageView(device, depthOnlyView, nullptr);

        PublishArmed();
        return;
    }

    SlotImages& slot = g_obj.slots[index];

    if (slot.depthOnly != VK_NULL_HANDLE)
        vkDestroyImageView(device, slot.depthOnly, nullptr);

    slot.depthOnly = depthOnlyView;

    RecordPass(g_obj, cmd, g_obj.motionPipeline, g_obj.sets[index][1], motionInfo.ImageView, slot.motion,
               regions.motion);

    if (frame.hasDepth)
        RecordPass(g_obj, cmd, g_obj.depthPipeline, g_obj.sets[index][0], depthSource, slot.depth, regions.depth);

    PublishArmed();
    g_lastCapture.store(frame.captureEpoch, std::memory_order_relaxed);
    g_lastCaptureTime = std::chrono::steady_clock::now();
    g_refusal.clear();

    static bool saidFirst = false;

    if (!saidFirst)
    {
        saidFirst = true;
        LOG_INFO("DLSS-NR Vulkan finished picture: first capture, depth {}{}, motion {}, output {}x{}, render subrect "
                 "{}x{}, MV low resolution {}, depth inverted {}",
                 frame.hasDepth ? FormatName(depth->Resource.ImageViewInfo.Format) : std::string("none"),
                 frame.hasDepth && depthRead == GuideRead::DepthOnly ? " (its depth aspect)" : "",
                 FormatName(motionInfo.Format), frame.outputWidth, frame.outputHeight, frame.renderWidth,
                 frame.renderHeight, frame.mvLowRes, frame.depthInverted);
    }
}

TakeResult Take(VkDevice device, VkQueue presentQueue, uint64_t presentEpoch, uint32_t width, uint32_t height,
                Guides& out)
{
    std::lock_guard lock(g_mutex);

    if (g_obj.device == VK_NULL_HANDLE || g_obj.device != device)
        return TakeResult::None;

    size_t index = 0;
    bool reset = false;

    // Captures on the present's own queue are ordered before the copy in by the queue itself; one from another queue is
    // used only when its timeline value was reached (never a GPU wait across queues: the work behind it can depend on
    // this very present, as in DlssNr_FinishedReady.h)
    const auto ready = [&](const GuideRing::Slot& slot)
    {
        if (slot.queue == (uint64_t) (uintptr_t) presentQueue)
            return true;

        if (slot.value == 0)
            return false; // no timeline: nothing says it has finished

        const auto it = g_obj.timelines.find(slot.queue);
        return it != g_obj.timelines.end() && it->second.semaphore != VK_NULL_HANDLE &&
               VulkanHooks::WaitTimelineSemaphore(device, it->second.semaphore, slot.value, 0);
    };

    const auto picked =
        g_ring.Pick(presentEpoch, g_snapshot.load(std::memory_order_relaxed), width, height, index, reset, ready);

    if (picked == PickResult::SizeMismatch)
    {
        g_refusal = "The upscaler output (" + std::to_string(g_ring.mismatchWidth) + "x" +
                    std::to_string(g_ring.mismatchHeight) + ") differs from the screen (" + std::to_string(width) +
                    "x" + std::to_string(height) + ").";
        return TakeResult::None;
    }

    // A capture that stays on another queue and not finished present after present: the game's DLSS and its present do
    // not order each other in a way this can use. Said once the run is long enough not to be a hiccup.
    static uint32_t notReadyRun = 0;

    if (picked == PickResult::NotReady)
    {
        if (++notReadyRun >= kNotReadyRefuse)
        {
            g_refusal = "The game's DLSS runs on another queue than its present: not supported yet.";
            static bool said = false;

            if (!said)
            {
                said = true;
                LOG_WARN(
                    "DLSS-NR Vulkan finished picture: the game's DLSS capture is on queue {:X}, its present on queue "
                    "{:X}, and the capture has not finished for {} presents in a row; refused",
                    (size_t) g_ring.At(index).queue, (size_t) (uintptr_t) presentQueue, notReadyRun);
            }
        }

        return TakeResult::NotReady;
    }

    notReadyRun = 0;

    if (picked != PickResult::Ready)
        return TakeResult::None;

    const FinishedGameFrame frame = g_ring.At(index).frame;
    const SlotImages& slot = g_obj.slots[index];
    out = Guides {};
    out.frame = frame;
    out.frame.reset = reset;
    out.motion = slot.motion.image;
    out.motionWidth = g_obj.motionWidth;
    out.motionHeight = g_obj.motionHeight;

    if (frame.hasDepth && slot.depth.image != VK_NULL_HANDLE)
    {
        out.depth = slot.depth.image;
        out.depthWidth = g_obj.depthWidth;
        out.depthHeight = g_obj.depthHeight;
    }
    else
        out.frame.hasDepth = false;

    return out.motion != VK_NULL_HANDLE ? TakeResult::Taken : TakeResult::None;
}

void BeginPresent() { g_snapshot.store(g_submitClock.load(std::memory_order_relaxed), std::memory_order_relaxed); }

void SetUsable(bool usable) { g_usable.store(usable, std::memory_order_relaxed); }

bool Recent(uint64_t presentCount)
{
    const uint64_t last = g_lastAttempt.load(std::memory_order_relaxed);
    return last != UINT64_MAX && presentCount >= last && presentCount - last <= kRecentPresents;
}

std::string Refusal()
{
    std::lock_guard lock(g_mutex);
    return g_refusal;
}

void OnPresent(uint64_t presentCount)
{
    // Nothing to do while nothing was ever captured, and without taking the lock
    const uint64_t last = g_lastCapture.load(std::memory_order_relaxed);

    if (last == UINT64_MAX)
        return;

    std::lock_guard lock(g_mutex);
    g_ring.Expire(presentCount);
    PublishArmed();

    if (g_obj.device == VK_NULL_HANDLE)
        return;

    DestroyRetired(g_obj, false, presentCount);

    // The game stopped calling its DLSS (a menu, Finished Picture or NR switched off, another upscaler): after a while
    // the ring and the pass are freed. Their last use is that many frames behind, so no wait is needed.
    if (presentCount > last && presentCount - last > kIdlePresents &&
        std::chrono::steady_clock::now() - g_lastCaptureTime > kIdleTime)
    {
        LOG_INFO("DLSS-NR Vulkan finished picture: no capture for {} presents, the capture images are released",
                 presentCount - last);
        Destroy(g_obj);
        g_ring.Clear();
        PublishArmed();
        g_lastCapture.store(UINT64_MAX, std::memory_order_relaxed);
        g_lastAttempt.store(UINT64_MAX, std::memory_order_relaxed);
    }
}

namespace
{
// The game's submit of `buffers` on `queue` succeeded: the captures recorded into them will run. A capture's work is
// ordered for a present on the same queue by the queue; for another queue an empty submit right behind it signals a
// timeline value (submit order puts every earlier batch of the queue before the signal).
void AfterSubmitBuffers(VkQueue queue, const VkCommandBuffer* buffers, uint32_t count)
{
    // Under the lock throughout, so the value taken and the signal queued stay in order (values on a queue never go
    // backwards); the nested vkQueueSubmit comes back through the submit hook with no buffers and does not lock
    std::lock_guard lock(g_mutex);

    if (g_obj.device == VK_NULL_HANDLE)
        return;

    bool waiting = false;

    for (uint32_t i = 0; i < count && !waiting; ++i)
        waiting = g_ring.Waiting(Handle(buffers[i]));

    VkDevice queueDevice = VK_NULL_HANDLE;
    VkPhysicalDevice queuePhysical = VK_NULL_HANDLE;

    if (!waiting || !native::VkFrameSource::DeviceOfQueue(queue, &queueDevice, &queuePhysical) ||
        queueDevice != g_obj.device)
        return;

    const uint64_t serial = g_submitClock.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t queueId = (uint64_t) (uintptr_t) queue;
    Objects::QueueTimeline* timeline = TimelineFor(queueId);
    const uint64_t value = timeline != nullptr ? timeline->value + 1 : 0;

    for (uint32_t i = 0; i < count; ++i)
        g_ring.Submitted(Handle(buffers[i]), queueId, serial, value);

    PublishArmed();

    if (timeline == nullptr)
        return;

    VkTimelineSemaphoreSubmitInfo values {};
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &value;
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &timeline->semaphore;
    const VkResult result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);

    if (result == VK_SUCCESS)
    {
        timeline->value = value;
        return;
    }

    // The signal will never come: those captures are not used (and reuse does not wait for them)
    g_ring.DropByValue(queueId, value);
    PublishArmed();
    static bool said = false;

    if (!said)
    {
        said = true;
        LOG_WARN("DLSS-NR Vulkan finished picture: the timeline signal behind the game's submit failed ({}); that "
                 "capture is dropped",
                 (int) result);
    }
}
} // namespace

void AfterSubmit(VkQueue queue, const VkCommandBuffer* buffers, uint32_t count)
{
    if (g_armed.load(std::memory_order_relaxed) == 0 || buffers == nullptr || count == 0)
        return;

    AfterSubmitBuffers(queue, buffers, count);
}

void AfterSubmit2(VkQueue queue, const VkCommandBufferSubmitInfo* buffers, uint32_t count)
{
    if (g_armed.load(std::memory_order_relaxed) == 0 || buffers == nullptr || count == 0)
        return;

    // A stack buffer, in chunks: no heap in the game's submit
    VkCommandBuffer handles[16];

    for (uint32_t first = 0; first < count; first += 16)
    {
        const uint32_t n = (std::min) (16u, count - first);

        for (uint32_t i = 0; i < n; ++i)
            handles[i] = buffers[first + i].commandBuffer;

        AfterSubmitBuffers(queue, handles, n);
    }
}

void ForgetCommandBuffers(uint32_t count, const VkCommandBuffer* buffers)
{
    if (g_armed.load(std::memory_order_relaxed) == 0 || buffers == nullptr)
        return;

    std::lock_guard lock(g_mutex);

    for (uint32_t i = 0; i < count; ++i)
        g_ring.Forget(Handle(buffers[i]));

    PublishArmed();
}

void NoteExecuteCommands(VkCommandBuffer primary, uint32_t count, const VkCommandBuffer* secondaries)
{
    if (g_armed.load(std::memory_order_relaxed) == 0 || secondaries == nullptr)
        return;

    std::lock_guard lock(g_mutex);

    for (uint32_t i = 0; i < count; ++i)
        g_ring.Rearm(Handle(secondaries[i]), Handle(primary));
}

void OnDeviceDestroyed(VkDevice device)
{
    std::lock_guard lock(g_mutex);

    if (g_obj.device == VK_NULL_HANDLE || g_obj.device != device)
        return;

    Destroy(g_obj);
    g_ring.Clear();
    PublishArmed();
    g_lastCapture.store(UINT64_MAX, std::memory_order_relaxed);
}
} // namespace DlssNr::FinishedVk
