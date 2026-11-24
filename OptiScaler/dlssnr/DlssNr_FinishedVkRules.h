#pragma once

// The rules of Finished Picture on a native Vulkan game (dlssnr/DlssNr_FinishedVk.h), apart from any Vulkan object so a
// host test can run them (tests/nr_finished_vk_rules_smoke.cpp): which formats of the game's depth and motion images
// the capture pass can read, and the bookkeeping of the capture ring -- which slots hold a frame, whether the game's
// command buffer that wrote one was really submitted, and which slot a present uses.
//
// The D3D12 Finished Picture knows when the game executes the command list that holds its capture
// (FinishedPictureSubmitted, DlssNr_Late.inl). On Vulkan the same knowledge comes from the vkQueueSubmit hooks: a slot
// is "armed" with the command buffer the capture was recorded into, becomes "submitted" when that command buffer is in
// a submit, and is dropped when the command buffer is begun again or freed first (the game recorded something else into
// it), or when it stays unsent for more than a few presents. A present only ever uses a submitted slot: a recording
// that never ran leaves no stale guides.

#include "DlssNr_FinishedGameFrame.h"
#include "../shaders/dlssnr/DlssNr_FinishedReady.h"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace DlssNr::FinishedVkRules
{
// How the capture pass reads a guide image through the view the game's upscaler call handed over.
enum class GuideRead
{
    Refused,  // not a format the pass can sample as floats
    View,     // the game's own view is read as it is (colour formats, depth-only formats)
    DepthOnly // a depth+stencil view: the pass makes its own depth-aspect view
};

// The motion vectors: two or four float or normalised channels, the first two are the vector.
inline GuideRead ReadMotion(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return GuideRead::View;
    default:
        return GuideRead::Refused;
    }
}

// The depth: a depth format (its depth aspect), or a single float or normalised colour channel.
inline GuideRead ReadDepth(VkFormat format, VkImageAspectFlags viewAspect)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
        return GuideRead::View;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        // A view that covers stencil too cannot be sampled; a depth-aspect-only one can
        return viewAspect == VK_IMAGE_ASPECT_DEPTH_BIT ? GuideRead::View : GuideRead::DepthOnly;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
        return GuideRead::View;
    default:
        return GuideRead::Refused;
    }
}

// Slots of the capture ring. A slot's descriptor set and depth-only view are rewritten when it is captured into again,
// and a command buffer the game recorded with it may still be waiting to run, so a slot is reused only once its capture
// is kReuseAge presents old (the game has at most three frames in flight), has no unsent recording and, where the
// device has timeline semaphores, the GPU has finished the submit that ran it; when no slot is free the capture is
// skipped. Without timelines the age alone stands for GPU completion and assumes the game keeps fewer than kReuseAge
// frames in flight. (No push descriptors: they need VK_KHR_push_descriptor on the game's device, which the game did not
// enable and the device hook does not add.)
inline constexpr size_t kSlots = 8;
inline constexpr uint64_t kReuseAge = 4;

// Presents an unsent recording may wait before it counts as never submitted. A game submits well before its present, so
// this only has to outlast a game that records a frame ahead.
inline constexpr uint64_t kArmedWindow = 6;

// The colour spaces of a Vulkan swapchain the finished picture can take: what VkFrameSource::ToColorSpace maps (sRGB
// non-linear as SDR, HDR10 ST 2084 as PQ, extended sRGB linear as scRGB). Anything else (Display P3, HLG, Dolby Vision,
// BT.709/BT.2020 linear, Adobe RGB, ...) would be decoded as sRGB, so it is refused by name.
inline bool ColourSpaceTaken(VkColorSpaceKHR space)
{
    return space == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR || space == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
           space == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
}

enum class PickResult
{
    None,        // nothing submitted that fits
    Ready,       // `index` may be composed now
    NotReady,    // the newest fitting capture is on another queue and has not finished yet
    SizeMismatch // captures are waiting, but for another output size (mismatchWidth/Height)
};

// The slots' bookkeeping. Callers lock; nothing in here is thread safe.
class GuideRing
{
  public:
    struct Slot
    {
        uint64_t serial = 0;       // order of capture; 0: never used
        uint64_t cmd = 0;          // the command buffer the capture was recorded into, while it is unsent
        bool armed = false;        // recorded, not yet seen in a submit
        bool submitted = false;    // seen in a successful submit, not yet used or dropped
        uint64_t armedAt = 0;      // the present count at the recording (for Expire)
        uint64_t queue = 0;        // the queue it was submitted on
        uint64_t submitSerial = 0; // order of the submit, to tell it from submits made after a present began
        uint64_t value = 0;        // the timeline value signalled on `queue` after the submit (0: none)
        FinishedGameFrame frame {};
    };

    uint32_t mismatchWidth = 0, mismatchHeight = 0; // set with PickResult::SizeMismatch

    // A capture is recorded into `cmd` at present count `now`: picks the slot to write (the oldest reusable one) and
    // returns false when none is free. A game that calls its upscaler twice in one command buffer (two views, two
    // features) gets two slots, both sent by the one submit; a command buffer recorded over is the begin/reset hook's
    // to say (Forget), not told apart here.
    bool Begin(uint64_t cmd, uint64_t now, const FinishedGameFrame& frame, size_t& index)
    {
        return Begin(cmd, now, frame, index, [](const Slot&) { return true; });
    }

    // `gpuDone(slot)`: the GPU has finished the work of the slot's last submit (always true for a slot with no timeline
    // value).
    template <class DoneFn>
    bool Begin(uint64_t cmd, uint64_t now, const FinishedGameFrame& frame, size_t& index, DoneFn&& gpuDone)
    {
        const Slot* best = nullptr;

        for (size_t i = 0; i < kSlots; ++i)
        {
            const Slot& slot = _slots[i];
            const bool free = slot.serial == 0 || (!slot.armed && now >= slot.frame.captureEpoch &&
                                                   now - slot.frame.captureEpoch >= kReuseAge && gpuDone(slot));

            if (free && (best == nullptr || slot.serial < best->serial))
            {
                best = &slot;
                index = i;
            }
        }

        if (best == nullptr)
            return false;

        Slot& slot = _slots[index];

        if (slot.submitted)
            Drop(slot);

        slot.serial = ++_serial;
        slot.cmd = cmd;
        slot.armed = true;
        slot.submitted = false;
        slot.armedAt = now;
        slot.queue = 0;
        slot.submitSerial = 0;
        slot.value = 0;
        slot.frame = frame;
        return true;
    }

    // The game's submit carrying `cmd` on `queue` returned success: what was recorded into it will run. `serial` is the
    // submit's number, `value` the timeline value the caller signals on `queue` right after (0: none). True when a slot
    // was waiting for it.
    bool Submitted(uint64_t cmd, uint64_t queue, uint64_t serial, uint64_t value)
    {
        bool any = false;

        for (auto& slot : _slots)
        {
            if (slot.armed && slot.cmd == cmd)
            {
                slot.armed = false;
                slot.submitted = true;
                slot.cmd = 0;
                slot.queue = queue;
                slot.submitSerial = serial;
                slot.value = value;
                any = true;
            }
        }

        return any;
    }

    // A recording into `cmd` waits for its submit.
    bool Waiting(uint64_t cmd) const
    {
        for (const auto& slot : _slots)
        {
            if (slot.armed && slot.cmd == cmd)
                return true;
        }

        return false;
    }

    // The signal that should have followed the submit with timeline `value` on `queue` could not be queued: those slots
    // would wait for it forever, so they are dropped (and report no value, so reuse does not wait for it either).
    bool DropByValue(uint64_t queue, uint64_t value)
    {
        bool any = false;

        for (auto& slot : _slots)
        {
            if (slot.submitted && slot.queue == queue && slot.value == value)
            {
                Drop(slot);
                slot.value = 0;
                any = true;
            }
        }

        return any;
    }

    // `cmd` is begun again, reset or freed: an unsent capture in it is gone.
    bool Forget(uint64_t cmd)
    {
        bool any = false;

        for (auto& slot : _slots)
        {
            if (slot.armed && slot.cmd == cmd)
            {
                Drop(slot);
                any = true;
            }
        }

        return any;
    }

    // A secondary command buffer was executed from `primary`: the submit will carry the primary.
    void Rearm(uint64_t secondary, uint64_t primary)
    {
        for (auto& slot : _slots)
        {
            if (slot.armed && slot.cmd == secondary)
                slot.cmd = primary;
        }
    }

    // Recordings that stayed unsent for more than kArmedWindow presents.
    void Expire(uint64_t now)
    {
        for (auto& slot : _slots)
        {
            if (slot.armed && now > slot.armedAt && now - slot.armedAt > kArmedWindow)
                Drop(slot);
        }
    }

    uint32_t Armed() const
    {
        uint32_t count = 0;

        for (const auto& slot : _slots)
            count += slot.armed ? 1 : 0;

        return count;
    }

    // The slot a present at epoch `epoch` composes on a picture of `width` x `height`: the newest capture of that size
    // that was submitted before the present began (`snapshot`: the submit clock when the present hook was entered; a
    // submit after it is for a later frame and stays), inside the window of FinishedStale (one present; the rule the
    // D3D12 slots use). `ready(slot)` says whether the work that wrote it is done or ordered by the present's own
    // queue: not ready, nothing is consumed and the present is skipped. Everything older than the one used goes with
    // it, and a game reset flag of a capture that was dropped unapplied rides on the one that is used. Captures are
    // matched by output size alone: a game that draws several views of the same size per frame (a multi-monitor layout)
    // gets the newest one's guides for the picture, as the D3D12 Finished Picture stands down for several views
    // altogether.
    template <class ReadyFn>
    PickResult Pick(uint64_t epoch, uint64_t snapshot, uint32_t width, uint32_t height, size_t& index, bool& reset,
                    ReadyFn&& ready)
    {
        const Slot* best = nullptr;
        bool otherSize = false;

        for (size_t i = 0; i < kSlots; ++i)
        {
            Slot& slot = _slots[i];

            if (!slot.submitted || slot.submitSerial > snapshot)
                continue;

            if (DlssNr::FinishedStale(epoch, slot.frame.captureEpoch, 1))
            {
                Drop(slot);
                continue;
            }

            if (slot.frame.outputWidth != width || slot.frame.outputHeight != height)
            {
                otherSize = true;
                mismatchWidth = slot.frame.outputWidth;
                mismatchHeight = slot.frame.outputHeight;
                continue;
            }

            if (best == nullptr || slot.serial > best->serial)
            {
                best = &slot;
                index = i;
            }
        }

        if (best == nullptr)
            return otherSize ? PickResult::SizeMismatch : PickResult::None;

        if (!ready(*best))
            return PickResult::NotReady;

        reset = best->frame.reset || _carriedReset;

        for (auto& slot : _slots)
        {
            if (slot.submitted && slot.submitSerial <= snapshot && slot.serial <= best->serial)
            {
                if (&slot != best && slot.frame.reset)
                    reset = true;

                slot.submitted = false;
            }
        }

        _carriedReset = false;
        return PickResult::Ready;
    }

    // Everything is forgotten (a resize, the device going): no capture of before is used again.
    void Clear()
    {
        for (auto& slot : _slots)
            slot = Slot {};

        _carriedReset = true;
    }

    const Slot& At(size_t index) const { return _slots[index]; }

  private:
    void Drop(Slot& slot)
    {
        // An unapplied capture's reset request is not lost: the next one applied carries it
        if ((slot.armed || slot.submitted) && slot.frame.reset)
            _carriedReset = true;

        slot.armed = false;
        slot.submitted = false;
        slot.cmd = 0;
    }

    std::array<Slot, kSlots> _slots {};
    uint64_t _serial = 0;
    size_t _next = 0;
    bool _carriedReset = false;
};

// What frame generation does to Finished Picture on a native Vulkan game. FSR FG and DLSS-G own the swapchain and
// replace or follow the game's present with frames of their own: not supported yet. F5Low (framegen/f5low/F5Low_Vk)
// keeps the game's own swapchain and takes the real picture from the present after Finished Picture has written its
// result into it (the present's wait list is the Finished Picture copy-back's semaphore, so the copy-in waits for it on
// the GPU); the generated frames are made from the finished pictures, as on D3D12. The game's own DLSS-G still blocks
// it.
enum class FgBlock
{
    None,
    OptiFg,   // FSR FG or DLSS-G output, active
    GameDlssg // the game's own DLSS-G is generating
};

inline FgBlock BlockedByFrameGeneration(bool optiFgActive, bool f5lowOutput, bool gameDlssgGenerates)
{
    if (gameDlssgGenerates)
        return FgBlock::GameDlssg;

    return optiFgActive && !f5lowOutput ? FgBlock::OptiFg : FgBlock::None;
}

inline const char* FgBlockText(FgBlock block)
{
    switch (block)
    {
    case FgBlock::OptiFg:
        return "Finished Picture does not run with FSR FG or DLSS-G output on a Vulkan game yet (F5Low works). Switch "
               "the FG "
               "output to F5Low or off, or pick After Super Resolution.";
    case FgBlock::GameDlssg:
        return "Finished Picture does not run while the game's own DLSS frame generation is on. Turn the game's frame "
               "generation off, or pick After Super Resolution.";
    default:
        return "";
    }
}

// The idle release of the private D3D12 pass (native/NativeDriverVk.cpp) shares that pass with Optical F5Low's NR (the
// same model, scratch pair and history). It may start only when Finished Picture has been idle for `limitMs` AND
// nothing else uses the pass: the motion estimate's NR (NativeInput with NR and Finished Picture on), the virtual
// upscaler, or a frame of it that ran lately. While another user is there the caller restarts the idle clock, so the
// release comes `limitMs` after the last use, long after that user's last list finished.
inline bool MayReleaseIdlePass(double nowMs, double lastActiveMs, double limitMs, bool otherNrUser)
{
    return !otherNrUser && nowMs - lastActiveMs >= limitMs;
}

// A capture made on another queue than the present's that stays not finished for this many presents in a row means the
// game's DLSS and its present are on queues that do not order each other here: refused with a status line.
inline constexpr uint32_t kNotReadyRefuse = 30;

// NR drops its history when nothing was applied yet or the capture of this frame is more than two game frames of
// presents after the last applied one (the D3D12 rule, one present per game frame: FinishedHistoryGap).
inline bool HistoryGap(uint64_t lastAppliedEpoch, uint64_t captureEpoch)
{
    return DlssNr::FinishedHistoryGap(lastAppliedEpoch, captureEpoch, 1);
}
} // namespace DlssNr::FinishedVkRules
