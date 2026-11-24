#pragma once

// Superior Scaling on native Vulkan: the dispatches of its No DLSS composition (SPIR-V built by
// tools/compile_superior_vk.ps1 from the same sources as the D3D12 blobs). What the frame does with them is
// dlssnr/DlssNr_Superior_Vk.inl.
//
// One descriptor layout serves every pass, the one of the composition pass (DlssNr_Vk): binding 0 the constants, 1-4
// four sampled reads (t0-t3), 5-6 two storage images (u0-u1), 7 the linear sampler, 8 the read t5 (the low field of the
// resolve) and 9 the read t4. Each blob uses the bindings it declares, as numbered in its source under VK_MODE, and
// leaves the rest.
//
// As the other Vulkan passes: every binding is written every dispatch (a slot a pass does not use gets a 1x1
// placeholder), constants and descriptor sets come from a ring of slots so the dispatches of one frame and of the
// frames still in flight never share one, the layouts of the images are the caller's to state, and each dispatch is
// followed by a compute-to-compute barrier. Compute and nothing else: no blits, no clears, no draws, so it is legal on
// the async-compute queue a game may present from.

#include "SysUtils.h"
#include <shaders/Shader_Vk.h>

#include <array>

class DlssNrSuperior_Vk : public Shader_Vk
{
  public:
    enum class Blob : uint32_t
    {
        Area,                         // v59 Area downsample of the encoded proxy to the model's raster
        SurfaceGuides,                // depth and motion matched to the model's raster, no depth
        JointGuides,                  // ... with depth
        JointMinifiedGuides,          // the correction pass after JointGuides where the guides shrink
        JointGuidesQuantized,         // JointGuides for a 16/24-bit integer depth
        JointMinifiedGuidesQuantized, // its correction pass
        LowRaw,                       // the 1/8-size low field of the model's answer minus its input
        ResolveGuidedRaw,             // the final composition of a raw answer, shaped by the low field
        // The private DLSS (CNN, Transformer):
        Generic,       // the v59 pass: Mode 7, the private DLSS's unit exposure
        CarrierDecode, // Mode 38: the DLSS output back to a signed residual
        LowSigned,     // Mode 18, Transfer 1: the low field of the signed residual
        ResolveHdr,    // the final composition of a signed residual (generic colour mapping, HDR / SDR)
        ResolveSdr,
        NormalSdr, // ... specialized to the colour mapping in force
        NormalHdr0,
        NormalHdr2,
        NormalHdr4,
        NormalHdr6,
        PrivateEncode, // carrier (0.5 + answer - proxy) and the exact private motion
        PrivateFade,   // the carrier decode blended toward the current edit (the CNN's D/E switch and preset fade)
        CnnMotion,     // the CNN's motion probe: 576 vectors -> a moving count
        GhostClear,    // the Transformer's disocclusion field: clear, scatter (coarse 1), age (coarse 2), references,
        GhostCoarse1,  // dilation, the bias mask for the DLSS, the decode that seeds the rings, the in-place feather
        GhostCoarse2,
        GhostReference,
        GhostDilate,
        GhostBias,
        GhostDecodeSeed,
        GhostRingsInPlace,
        Count
    };

    // A sampled read: the view, and the layout its image is in when the dispatch runs. A null view binds the
    // placeholder.
    struct Read
    {
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    };
    // t0, t1, t2, t3 (bindings 1-4), t4 (binding 9) and t5 (binding 8): the index is the register number.
    static constexpr uint32_t kReads = 6;
    using Reads = std::array<Read, kReads>;

    // The dispatches of one frame: area, guides (one slot for the pair), the private DLSS's encode, ghost field (6),
    // decode, feather, low field and resolve come to 14 with the Transformer; sixteen leaves room. Six frames are twice
    // the three that can be in flight, with no fence, on the assumption of one NR evaluate per rendered frame.
    static constexpr uint32_t kSlotsPerFrame = 16;
    static constexpr uint32_t kFramesInFlight = 6;
    static constexpr uint32_t kSlots = kSlotsPerFrame * kFramesInFlight;

    DlssNrSuperior_Vk(std::string InName, VkDevice InDevice, VkPhysicalDevice InPhysicalDevice);
    ~DlssNrSuperior_Vk();

    // The No DLSS composition's passes are built and usable.
    bool Ready() const { return _init && !_baseFailed; }
    // The private DLSS's passes did not fail to build (their failure leaves Ready() alone: Superior then runs No DLSS).
    bool PrivateReady() const { return !_privateFailed; }

    // Whether `blob` belongs to the private DLSS (its failure does not give the pass up).
    static constexpr bool IsPrivate(Blob blob) { return blob >= Blob::Generic; }

    // Builds the pipelines of `blobs` and the placeholder image now, so a build that fails is known before any frame
    // depends on them (the pass is then given up for the selection, not a frame's dispatch). False: one would not
    // build.
    bool Build(VkCommandBuffer InCmdList, const Blob* InBlobs, size_t InCount);

    // The colour-mapping specializations of the residual resolve are an optimization of the generic resolve: one that
    // would not build is skipped (the generic one runs) and does not give the pass up.
    static constexpr bool Optional(Blob blob) { return blob >= Blob::NormalSdr && blob <= Blob::NormalHdr6; }

    // Whether this device can run the low field pass (its tile needs 29,376 bytes of shared memory).
    static bool DeviceCanRun(VkPhysicalDevice physicalDevice);

    // One dispatch of `blob`. InConstants is copied into the pass's slot (at most 512 bytes: the Superior constants are
    // the first 256 of the live struct, which is 512); the written images are InOut0 (u0, binding 5) and InOut1 (u1,
    // binding 6) in GENERAL. groupsX x groupsY are thread groups: the callers know each blob's group size.
    bool Dispatch(VkCommandBuffer InCmdList, Blob InBlob, const void* InConstants, size_t InConstantBytes,
                  uint32_t InGroupsX, uint32_t InGroupsY, const Reads& InReads, VkImageView InOut0,
                  VkImageView InOut1 = VK_NULL_HANDLE);

    // The guide pair: `InPrimary` over groupsX x groupsY, then, if InCorrection is not Count, the correction blob over
    // its own groups, both on one set of constants and bindings and with a barrier between.
    bool DispatchGuides(VkCommandBuffer InCmdList, Blob InPrimary, Blob InCorrection, const void* InConstants,
                        size_t InConstantBytes, uint32_t InGroupsX, uint32_t InGroupsY, uint32_t InCorrectionX,
                        uint32_t InCorrectionY, const Reads& InReads, VkImageView InOut0, VkImageView InOut1);

  private:
    VkDeviceSize _slotStride = 0;
    uint32_t _slot = 0;

    std::array<VkPipeline, (size_t) Blob::Count> _blobs {};
    std::array<bool, (size_t) Blob::Count> _blobTried {};
    bool _baseFailed = false; // a No DLSS blob that could not be built: the pass is given up, not retried every frame
    bool _privateFailed = false; // a private DLSS blob that could not: the private DLSS is given up (No DLSS runs)

    VkImage _dummyImage = VK_NULL_HANDLE;
    VkDeviceMemory _dummyMemory = VK_NULL_HANDLE;
    VkImageView _dummyView = VK_NULL_HANDLE;
    bool _dummyReady = false;
    bool _dummyFailed = false; // not tried again: a half-made placeholder is never used

    bool CreateDummy(VkCommandBuffer cmdList);
    VkPipeline PipelineOf(Blob blob);
    // Takes a slot, writes the constants and the descriptors; returns the slot's set (null on failure).
    VkDescriptorSet Prepare(VkCommandBuffer cmdList, const void* constants, size_t bytes, const Reads& reads,
                            VkImageView out0, VkImageView out1);
    void Record(VkCommandBuffer cmdList, VkPipeline pipeline, VkDescriptorSet set, uint32_t groupsX, uint32_t groupsY);
};
