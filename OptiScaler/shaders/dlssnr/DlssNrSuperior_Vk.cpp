#include "pch.h"

#include "DlssNrSuperior_Vk.h"

#include "DlssNr_Common.h"
#include <dlssnr/DlssNr_SuperiorVkRule.h>

#include "superior/precompile_vk/dlssnr_superior_area_Shader_Vk.h"
#include "superior/precompile_vk/dlssnr_superior_carrier_Shader_Vk.h"
#include "superior/precompile_vk/dlssnr_superior_generic_Shader_Vk.h"
#include "superior/precompile_vk/dlssnr_superior_low_Shader_Vk.h"
#include "superior/precompile_vk/host_normal_hdr0_Shader_Vk.h"
#include "superior/precompile_vk/host_normal_hdr2_Shader_Vk.h"
#include "superior/precompile_vk/host_normal_hdr4_Shader_Vk.h"
#include "superior/precompile_vk/host_normal_hdr6_Shader_Vk.h"
#include "superior/precompile_vk/host_normal_sdr_Shader_Vk.h"
#include "superior/precompile_vk/host_resolve_hdr_Shader_Vk.h"
#include "superior/precompile_vk/host_resolve_sdr_Shader_Vk.h"
#include "superior/precompile_vk/private_cnn_motion_Shader_Vk.h"
#include "superior/precompile_vk/private_encode_Shader_Vk.h"
#include "superior/precompile_vk/private_fade_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_bias_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_clear_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_coarse_1_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_coarse_2_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_decode_seed_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_dilate_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_reference_Shader_Vk.h"
#include "superior/precompile_vk/private_ghost_rings_inplace_Shader_Vk.h"
#include "superior/precompile_vk/host_low_raw_Shader_Vk.h"
#include "superior/precompile_vk/host_resolve_guided_raw_Shader_Vk.h"
#include "superior/precompile_vk/host_uniform_joint_guides_Shader_Vk.h"
#include "superior/precompile_vk/host_uniform_joint_guides_quantized_Shader_Vk.h"
#include "superior/precompile_vk/host_uniform_joint_minified_guides_Shader_Vk.h"
#include "superior/precompile_vk/host_uniform_joint_minified_guides_quantized_Shader_Vk.h"
#include "superior/precompile_vk/host_uniform_surface_guides_Shader_Vk.h"

#include <algorithm>
#include <cstring>

namespace
{
struct BlobCode
{
    const unsigned char* data;
    size_t size;
    const char* name;
};

BlobCode CodeOf(DlssNrSuperior_Vk::Blob blob)
{
    using B = DlssNrSuperior_Vk::Blob;
    switch (blob)
    {
    case B::Area:
        return { dlssnr_superior_area_spv, sizeof(dlssnr_superior_area_spv), "Area downsample" };
    case B::SurfaceGuides:
        return { host_uniform_surface_guides_spv, sizeof(host_uniform_surface_guides_spv), "surface guides" };
    case B::JointGuides:
        return { host_uniform_joint_guides_spv, sizeof(host_uniform_joint_guides_spv), "joint guides" };
    case B::JointMinifiedGuides:
        return { host_uniform_joint_minified_guides_spv, sizeof(host_uniform_joint_minified_guides_spv),
                 "minified joint guides" };
    case B::JointGuidesQuantized:
        return { host_uniform_joint_guides_quantized_spv, sizeof(host_uniform_joint_guides_quantized_spv),
                 "quantized joint guides" };
    case B::JointMinifiedGuidesQuantized:
        return { host_uniform_joint_minified_guides_quantized_spv,
                 sizeof(host_uniform_joint_minified_guides_quantized_spv), "minified quantized joint guides" };
    case B::LowRaw:
        return { host_low_raw_spv, sizeof(host_low_raw_spv), "low field" };
    case B::ResolveGuidedRaw:
        return { host_resolve_guided_raw_spv, sizeof(host_resolve_guided_raw_spv), "guided resolve" };
    case B::Generic:
        return { dlssnr_superior_generic_spv, sizeof(dlssnr_superior_generic_spv), "unit exposure" };
    case B::CarrierDecode:
        return { dlssnr_superior_carrier_spv, sizeof(dlssnr_superior_carrier_spv), "carrier decode" };
    case B::LowSigned:
        return { dlssnr_superior_low_spv, sizeof(dlssnr_superior_low_spv), "residual low field" };
    case B::ResolveHdr:
        return { host_resolve_hdr_spv, sizeof(host_resolve_hdr_spv), "residual resolve (HDR)" };
    case B::ResolveSdr:
        return { host_resolve_sdr_spv, sizeof(host_resolve_sdr_spv), "residual resolve (SDR)" };
    case B::NormalSdr:
        return { host_normal_sdr_spv, sizeof(host_normal_sdr_spv), "residual resolve (SDR mapping)" };
    case B::NormalHdr0:
        return { host_normal_hdr0_spv, sizeof(host_normal_hdr0_spv), "residual resolve (HDR mapping 0)" };
    case B::NormalHdr2:
        return { host_normal_hdr2_spv, sizeof(host_normal_hdr2_spv), "residual resolve (HDR mapping 2)" };
    case B::NormalHdr4:
        return { host_normal_hdr4_spv, sizeof(host_normal_hdr4_spv), "residual resolve (HDR mapping 4)" };
    case B::NormalHdr6:
        return { host_normal_hdr6_spv, sizeof(host_normal_hdr6_spv), "residual resolve (HDR mapping 6)" };
    case B::PrivateEncode:
        return { private_encode_spv, sizeof(private_encode_spv), "private encode" };
    case B::PrivateFade:
        return { private_fade_spv, sizeof(private_fade_spv), "private fade" };
    case B::CnnMotion:
        return { private_cnn_motion_spv, sizeof(private_cnn_motion_spv), "CNN motion probe" };
    case B::GhostClear:
        return { private_ghost_clear_spv, sizeof(private_ghost_clear_spv), "ghost clear" };
    case B::GhostCoarse1:
        return { private_ghost_coarse_1_spv, sizeof(private_ghost_coarse_1_spv), "ghost scatter" };
    case B::GhostCoarse2:
        return { private_ghost_coarse_2_spv, sizeof(private_ghost_coarse_2_spv), "ghost age" };
    case B::GhostReference:
        return { private_ghost_reference_spv, sizeof(private_ghost_reference_spv), "ghost reference" };
    case B::GhostDilate:
        return { private_ghost_dilate_spv, sizeof(private_ghost_dilate_spv), "ghost dilate" };
    case B::GhostBias:
        return { private_ghost_bias_spv, sizeof(private_ghost_bias_spv), "ghost bias mask" };
    case B::GhostDecodeSeed:
        return { private_ghost_decode_seed_spv, sizeof(private_ghost_decode_seed_spv), "ghost decode" };
    case B::GhostRingsInPlace:
        return { private_ghost_rings_inplace_spv, sizeof(private_ghost_rings_inplace_spv), "ghost feather" };
    default:
        return { nullptr, 0, "" };
    }
}
} // namespace

DlssNrSuperior_Vk::DlssNrSuperior_Vk(std::string InName, VkDevice InDevice, VkPhysicalDevice InPhysicalDevice)
    : Shader_Vk(InName, InDevice, InPhysicalDevice)
{
    if (InDevice == VK_NULL_HANDLE || InPhysicalDevice == VK_NULL_HANDLE)
    {
        _init = false;
        return;
    }

    // Linear clamp, as the D3D12 pass's s0.
    CreateSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(_physicalDevice, &props);

    // The resolve takes the live constants (512 bytes), every other pass the first 256 of a slot of the same size.
    const VkDeviceSize alignment = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    _slotStride = ((sizeof(::DlssNrConstants) + alignment - 1) / alignment) * alignment;

    if (!CreateBufferResource(_device, _physicalDevice, &_constantBuffer, &_constantBufferMemory, _slotStride * kSlots,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
        vkMapMemory(_device, _constantBufferMemory, 0, _slotStride * kSlots, 0, &_mappedConstantBuffer) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan Superior Scaling: could not allocate its constant ring");
        _init = false;
        return;
    }

    // The composition pass's layout (DlssNr_Vk), entry for entry: the blobs carry its binding numbers.
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        CreateBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
        CreateBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        CreateBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        CreateBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        CreateBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        CreateBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
        CreateBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
        CreateBinding(7, VK_DESCRIPTOR_TYPE_SAMPLER),
        CreateBinding(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        CreateBinding(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
    };

    CreateLayouts(bindings);

    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlots },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kReads * kSlots },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kSlots },
        { VK_DESCRIPTOR_TYPE_SAMPLER, kSlots },
    };

    CreateDescriptorPool(poolSizes, kSlots);

    // One set per slot (CreateDescriptorSets makes _maxFramesInFlight of them).
    _maxFramesInFlight = kSlots;
    CreateDescriptorSets(_descriptorSetLayout, _descriptorPool, _descriptorSets);
    _maxFramesInFlight = kFramesInFlight;

    bool setsOk = _descriptorSets.size() >= kSlots;
    for (const VkDescriptorSet set : _descriptorSets)
        setsOk = setsOk && set != VK_NULL_HANDLE;

    if (!setsOk || _descriptorSetLayout == VK_NULL_HANDLE || _pipelineLayout == VK_NULL_HANDLE ||
        _descriptorPool == VK_NULL_HANDLE || _textureSampler == VK_NULL_HANDLE)
    {
        LOG_ERROR("DLSS-NR Vulkan Superior Scaling: its descriptor sets, layouts or sampler could not be created");
        _init = false;
        return;
    }

    _init = true;
}

DlssNrSuperior_Vk::~DlssNrSuperior_Vk()
{
    if (_device == VK_NULL_HANDLE)
        return;

    for (VkPipeline& pipeline : _blobs)
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(_device, pipeline, nullptr);

    if (_dummyView != VK_NULL_HANDLE)
        vkDestroyImageView(_device, _dummyView, nullptr);

    if (_dummyImage != VK_NULL_HANDLE)
        vkDestroyImage(_device, _dummyImage, nullptr);

    if (_dummyMemory != VK_NULL_HANDLE)
        vkFreeMemory(_device, _dummyMemory, nullptr);
}

bool DlssNrSuperior_Vk::DeviceCanRun(VkPhysicalDevice physicalDevice)
{
    if (physicalDevice == VK_NULL_HANDLE)
        return false;

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    return DlssNr::SuperiorScaling::VkRule::SharedMemoryEnough(props.limits.maxComputeSharedMemorySize) &&
           props.limits.maxComputeWorkGroupInvocations >= 128;
}

// One RGBA16F pixel in GENERAL, legal for both a sampled read and a storage write; its content is never read. It stands
// in for every binding a pass does not use.
bool DlssNrSuperior_Vk::CreateDummy(VkCommandBuffer cmdList)
{
    if (_dummyReady)
        return true;

    if (_dummyFailed)
        return false;

    _dummyFailed = true;

    VkImageCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.extent = { 1, 1, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(_device, &info, nullptr, &_dummyImage) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req {};
    vkGetImageMemoryRequirements(_device, _dummyImage, &req);

    VkMemoryAllocateInfo alloc {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = FindMemoryType(_physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(_device, &alloc, nullptr, &_dummyMemory) != VK_SUCCESS ||
        vkBindImageMemory(_device, _dummyImage, _dummyMemory, 0) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo view {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = _dummyImage;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    if (vkCreateImageView(_device, &view, nullptr, &_dummyView) != VK_SUCCESS)
        return false;

    VkImageSubresourceRange range { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    SetImageLayout(cmdList, _dummyImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, range);

    _dummyFailed = false;
    _dummyReady = true;
    return true;
}

bool DlssNrSuperior_Vk::Build(VkCommandBuffer InCmdList, const Blob* InBlobs, size_t InCount)
{
    if (!Ready() || InCmdList == VK_NULL_HANDLE || !CreateDummy(InCmdList))
        return false;

    bool ok = true;
    for (size_t i = 0; i < InCount; ++i)
        ok = (PipelineOf(InBlobs[i]) != VK_NULL_HANDLE || Optional(InBlobs[i])) && ok;
    return ok;
}

VkPipeline DlssNrSuperior_Vk::PipelineOf(Blob blob)
{
    const size_t index = (size_t) blob;
    if (index >= _blobs.size())
        return VK_NULL_HANDLE;

    if (!_blobTried[index])
    {
        _blobTried[index] = true;
        const BlobCode code = CodeOf(blob);
        std::vector<char> bytes(code.data, code.data + code.size);

        if (!CreateComputePipeline(_device, _pipelineLayout, &_blobs[index], bytes) || _blobs[index] == VK_NULL_HANDLE)
        {
            _blobs[index] = VK_NULL_HANDLE;
            if (!Optional(blob))
                (IsPrivate(blob) ? _privateFailed : _baseFailed) = true;
            LOG_WARN("DLSS-NR Vulkan Superior Scaling: the {} pass could not be built", code.name);
        }
    }

    return _blobs[index];
}

VkDescriptorSet DlssNrSuperior_Vk::Prepare(VkCommandBuffer cmdList, const void* constants, size_t bytes,
                                           const Reads& reads, VkImageView out0, VkImageView out1)
{
    if (!Ready() || cmdList == VK_NULL_HANDLE || out0 == VK_NULL_HANDLE || constants == nullptr ||
        bytes > sizeof(::DlssNrConstants))
        return VK_NULL_HANDLE;

    if (!CreateDummy(cmdList))
    {
        LOG_ERROR("DLSS-NR Vulkan Superior Scaling: could not create its placeholder image");
        return VK_NULL_HANDLE;
    }

    const uint32_t slot = _slot;
    _slot = (_slot + 1) % kSlots;

    const VkDeviceSize offset = _slotStride * slot;
    char* const mapped = (char*) _mappedConstantBuffer + offset;
    std::memset(mapped, 0, sizeof(::DlssNrConstants));
    std::memcpy(mapped, constants, bytes);

    const VkDescriptorSet set = _descriptorSets[slot];
    VkDescriptorBufferInfo bufferInfo { _constantBuffer, offset, sizeof(::DlssNrConstants) };

    VkDescriptorImageInfo readInfo[kReads] {};
    for (uint32_t i = 0; i < kReads; ++i)
    {
        const bool given = reads[i].view != VK_NULL_HANDLE;
        readInfo[i] = { _textureSampler, given ? reads[i].view : _dummyView,
                        given ? reads[i].layout : VK_IMAGE_LAYOUT_GENERAL };
    }

    VkDescriptorImageInfo target0 { VK_NULL_HANDLE, out0, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo target1 { VK_NULL_HANDLE, out1 != VK_NULL_HANDLE ? out1 : _dummyView,
                                    VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sampler { _textureSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };

    // Bindings 0, 1-4 (t0-t3), 5-6, 7, 8 (t5), 9 (t4).
    VkWriteDescriptorSet writes[10] {};
    const uint32_t bindingOf[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (uint32_t i = 0; i < 10; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = bindingOf[i];
        writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &bufferInfo;
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1 + i].pImageInfo = &readInfo[i];
    }
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = &target0;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[6].pImageInfo = &target1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[7].pImageInfo = &sampler;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &readInfo[5];
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &readInfo[4];

    vkUpdateDescriptorSets(_device, 10, writes, 0, nullptr);
    return set;
}

void DlssNrSuperior_Vk::Record(VkCommandBuffer cmdList, VkPipeline pipeline, VkDescriptorSet set, uint32_t groupsX,
                               uint32_t groupsY)
{
    vkCmdBindPipeline(cmdList, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdList, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmdList, groupsX, groupsY, 1);

    // What this dispatch wrote, the next one reads.
    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdList, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
}

bool DlssNrSuperior_Vk::Dispatch(VkCommandBuffer InCmdList, Blob InBlob, const void* InConstants,
                                 size_t InConstantBytes, uint32_t InGroupsX, uint32_t InGroupsY, const Reads& InReads,
                                 VkImageView InOut0, VkImageView InOut1)
{
    if (InGroupsX == 0 || InGroupsY == 0)
        return false;

    const VkPipeline pipeline = PipelineOf(InBlob);
    if (pipeline == VK_NULL_HANDLE)
        return false;

    const VkDescriptorSet set = Prepare(InCmdList, InConstants, InConstantBytes, InReads, InOut0, InOut1);
    if (set == VK_NULL_HANDLE)
        return false;

    Record(InCmdList, pipeline, set, InGroupsX, InGroupsY);
    return true;
}

bool DlssNrSuperior_Vk::DispatchGuides(VkCommandBuffer InCmdList, Blob InPrimary, Blob InCorrection,
                                       const void* InConstants, size_t InConstantBytes, uint32_t InGroupsX,
                                       uint32_t InGroupsY, uint32_t InCorrectionX, uint32_t InCorrectionY,
                                       const Reads& InReads, VkImageView InOut0, VkImageView InOut1)
{
    if (InGroupsX == 0 || InGroupsY == 0 || InOut1 == VK_NULL_HANDLE)
        return false;

    const bool correct = InCorrection != Blob::Count;
    const VkPipeline primary = PipelineOf(InPrimary);
    const VkPipeline correction = correct ? PipelineOf(InCorrection) : VK_NULL_HANDLE;
    if (primary == VK_NULL_HANDLE ||
        (correct && (correction == VK_NULL_HANDLE || InCorrectionX == 0 || InCorrectionY == 0)))
        return false;

    const VkDescriptorSet set = Prepare(InCmdList, InConstants, InConstantBytes, InReads, InOut0, InOut1);
    if (set == VK_NULL_HANDLE)
        return false;

    Record(InCmdList, primary, set, InGroupsX, InGroupsY);
    if (correct)
        Record(InCmdList, correction, set, InCorrectionX, InCorrectionY);
    return true;
}
