#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// F5Low on a native Vulkan swapchain (framegen/f5low/F5Low_Vk.h): the pure counting and timing rules, host-tested in
// tests/f5low_vk_rules_smoke.cpp.
namespace F5LowVk
{
// Frames in flight of our own per-frame objects (command buffers, semaphores). A present that waits on one of our
// semaphores is only known to be done when the swapchain hands the image out again, so the ring is the size of a
// swapchain's usual image count.
inline constexpr uint32_t kSlots = 4;

// What the generator will be asked for.
inline constexpr uint32_t kMaxGenerated = 5;

// The game thread never waits longer than this in one present, in all (the job two frames back, the slot, the fences,
// the worker's drain, the swapchain lock). The worker's wait for the swapchain lock is longer: a game's direct present
// may block on vsync inside it.
inline constexpr double kBlockMsRule = 50.0;
inline constexpr unsigned long kWorkerLockMsRule = 1000;

// What is left of the game thread's budget (at least 1 ms, a zero wait would be a poll that fails by itself)
inline uint32_t RemainingMs(double budgetMs, double elapsedMs)
{
    const double left = budgetMs - elapsedMs;
    return left < 1.0 ? 1u : static_cast<uint32_t>(left);
}

// The swapchain's image count asked of the driver: the game's own plus one image per frame we may generate, capped at
// what the surface offers (surfaceMax 0 = no limit).
inline uint32_t RequestedMinImageCount(uint32_t gameMin, uint32_t extraImages, uint32_t surfaceMax)
{
    const uint32_t wanted = gameMin + extraImages;
    return surfaceMax == 0 ? wanted : std::min(wanted, std::max(surfaceMax, gameMin));
}

// How many images the driver gave beyond the game's minimum: the headroom we generate into.
inline uint32_t ImageHeadroom(uint32_t imageCount, uint32_t gameMin)
{
    return imageCount > gameMin ? imageCount - gameMin : 0;
}

// Frames to generate per game frame: what the multiplier asks for, never above the generator's limit nor the images
// we can acquire next to the game's.
inline uint32_t GeneratedCount(int requested, uint32_t generatorMax, uint32_t headroom)
{
    if (requested <= 0)
        return 0;

    return std::min({ static_cast<uint32_t>(requested), generatorMax, kMaxGenerated, headroom });
}

// Smoothed game frame interval from the time between two presents. A sample out of range (a pause, a hitch, the first
// frame) is dropped; 0 stays "unknown".
inline double NextInterval(double previousMs, double sampleMs)
{
    if (!(sampleMs >= 2.0 && sampleMs <= 100.0))
        return previousMs;

    if (previousMs <= 0.0)
        return sampleMs;

    return previousMs * 0.8 + sampleMs * 0.2;
}

// The game's own frame time: the time between two present arrivals minus what its thread spent in our blocking waits
// in between (the gate, the slot and fence waits, the inline presents, the swapchain lock, the game's acquire polling).
// Without that subtraction our waits lengthen the interval, which widens the spacing, which lengthens the waits.
inline double Unthrottled(double deltaMs, double ourWaitMs) { return deltaMs - (ourWaitMs > 0.0 ? ourWaitMs : 0.0); }

// Space between the generated frames' presents on the CPU: none when the display paces (V-Sync / FIFO: present back to
// back, the queue holds the order), else the interval less a margin, shared by the batch; a batch never outlasts the
// interval
inline double BatchSpacingMs(double intervalMs, uint32_t generated, bool displayPaced, double marginMs = 0.5)
{
    if (displayPaced || generated == 0 || !(intervalMs > marginMs))
        return 0.0;

    return (intervalMs - marginMs) / (generated + 1);
}

// Space between the shown frames of one game frame (the generated ones, then the game's own). The waits add up to
// count * spacing, which is always below one interval. 0 when the interval is not known: no pacing, no generation.
inline double SpacingMs(double intervalMs, uint32_t count)
{
    if (!(intervalMs > 0.0) || count == 0)
        return 0.0;

    return intervalMs / (count + 1);
}

// A fence the generator signals: true when its completed value is more than `allowed` values behind what was issued, or
// is UINT64_MAX (D3D12 says so for a removed device). The game queue must not wait on a value that will not arrive.
inline bool FenceBehind(uint64_t completed, uint64_t issued, uint64_t allowed = 1)
{
    if (completed == UINT64_MAX)
        return true;

    return issued > completed && issued - completed > allowed;
}

// A timeline value the host hands out: the next one is read before the submit and only taken once it succeeded, so a
// failed submit leaves no gap and the values strictly increase.
struct FenceCounter
{
    uint64_t issued = 0;
    uint64_t Peek() const { return issued + 1; }
    uint64_t Take() { return ++issued; }
};

// What the frame shows after f5low's answer (result as F5LowFramesResult: 0 OK; anything else makes no frames). f5low
// makes all asked frames or none.
inline uint32_t GeneratedAfter(int result, uint32_t reported, uint32_t asked)
{
    return result == 0 && reported >= asked ? asked : 0;
}

// A VkDeviceGroupPresentInfoKHR (RDR2 chains one) the real and the generated presents may carry: one device only.
// mode is its VkDeviceGroupPresentModeFlagsKHR (1 = LOCAL); with swapchainCount 1 the swapchain's device mask has one
// bit, with 0 the default applies.
inline bool SingleDeviceMask(uint32_t mask) { return mask != 0 && (mask & (mask - 1)) == 0; }

inline bool DeviceGroupCarryable(uint32_t swapchainCount, uint32_t mode, uint32_t mask0)
{
    if (mode != 1)
        return false;

    if (swapchainCount == 0)
        return true;

    return swapchainCount == 1 && SingleDeviceMask(mask0);
}

// A raise of the multiplier past what the swapchain was made for and past its spare images
inline bool NeedsRaise(int wanted, int createdFor, uint32_t headroom)
{
    return wanted > createdFor && wanted > static_cast<int>(headroom);
}

// Getting the game to make a new swapchain for a raise: idle -> asked (a SUBOPTIMAL present) -> forcePending (no new
// swapchain after kAskPresents presents and kAskMs) -> forced (its next acquire answered OUT_OF_DATE, once) -> gaveUp
// (no new swapchain kForcedMs later: the clamp and the status line stay). A new swapchain or another wanted count
// starts over; a lowered count (or enough spare images) goes back to idle.
enum class RecreateState
{
    Idle,
    Asked,
    ForcePending,
    Forced,
    GaveUp
};

inline constexpr uint32_t kAskPresents = 30;
inline constexpr double kAskMs = 500.0;
inline constexpr double kForcedMs = 2000.0;

struct RecreateAsk
{
    RecreateState state = RecreateState::Idle;
    int wanted = 0;
    uint32_t presents = 0;
    double sinceMs = 0.0;
};

// Per present that went through. Returns true when this present should answer SUBOPTIMAL to the game.
inline bool OnPresentRecreate(RecreateAsk& ask, int wanted, int createdFor, uint32_t headroom, double nowMs)
{
    if (ask.state != RecreateState::Idle && ask.wanted != wanted)
        ask = {};

    if (!NeedsRaise(wanted, createdFor, headroom))
    {
        ask = {};
        return false;
    }

    switch (ask.state)
    {
    case RecreateState::Idle:
        ask = { RecreateState::Asked, wanted, 0, nowMs };
        return true;
    case RecreateState::Asked:
        if (++ask.presents >= kAskPresents && nowMs - ask.sinceMs >= kAskMs)
            ask.state = RecreateState::ForcePending;
        return false;
    case RecreateState::Forced:
        if (nowMs - ask.sinceMs >= kForcedMs)
            ask.state = RecreateState::GaveUp;
        return false;
    default:
        return false;
    }
}

// The game's acquire on the swapchain: true once, when the escalation is due (answer OUT_OF_DATE, acquire nothing)
inline bool OnAcquireRecreate(RecreateAsk& ask, double nowMs)
{
    if (ask.state != RecreateState::ForcePending)
        return false;

    ask.state = RecreateState::Forced;
    ask.sinceMs = nowMs;
    return true;
}

// Spare images asked for up front: the most frames this session may generate, so the multiplier never needs a new
// swapchain. (RequestedMinImageCount caps it by the surface.)
inline uint32_t SpareImagesUpFront(int maxInterpolations)
{
    return static_cast<uint32_t>(std::clamp(maxInterpolations, 1, static_cast<int>(kMaxGenerated)));
}

// The spare images to ask for: the most the session may generate when the latency bound keeps them from adding queue
// depth, else only what the multiplier needs (and a recreate when it is raised).
inline uint32_t SpareImagesToAsk(bool bounded, int maxInterpolations, int wanted)
{
    return bounded ? SpareImagesUpFront(maxInterpolations) : SpareImagesUpFront(std::min(wanted, maxInterpolations));
}

// The game's latency bound: how many of its presents may be unshown at once. Without us it had gameMinImages images,
// one on screen and the rest in flight; never below 2 (a bound of 1 would serialise a game that is not generating), and
// 3 when one frame is generated (a batch of 2 images is shown in two refreshes: the game must be a batch ahead or the
// display waits for it) and 3 when the batch is timed on the CPU (MAILBOX / IMMEDIATE: the batch takes about a whole
// interval, so the game must be a whole batch ahead; closed-loop test: bound 2 holds a 60 fps game at 6X to 53 fps).
inline constexpr uint32_t kGateRing = 8; // present ids remembered
inline uint32_t LatencyBound(uint32_t gameMinImages, uint32_t generated, bool displayPaced)
{
    const uint32_t bound = gameMinImages > 1 ? gameMinImages - 1 : 1;
    const uint32_t floor = generated == 1 || !displayPaced ? 3u : 2u;
    return bound < floor ? floor : bound;
}
inline bool GateUsable(uint32_t bound) { return bound >= 1 && bound < kGateRing; }

// Before the game acquires for its next present (number issued + 1): the number (1-based) of the present that has to be
// on screen already, 0 when none yet.
inline uint64_t GateWaitNumber(uint64_t issued, uint32_t bound)
{
    const uint64_t next = issued + 1;
    return next > bound ? next - bound : 0;
}

// The gate stands aside after this many waits in a row that ran into their limit, and comes back after
// kGateRearmPresents presents or kGateRearmMs (a minimise or an Alt-Tab must not switch it off for good).
inline constexpr uint32_t kGateGiveUp = 5;
inline constexpr uint32_t kGateRearmPresents = 120;
inline constexpr double kGateRearmMs = 3000.0;

enum class WaitOutcome
{
    Done,
    Timeout,
    Neutral // OUT_OF_DATE / SURFACE_LOST: the swapchain is going, says nothing about the gate
};

struct GateState
{
    bool on = true;
    uint32_t timeouts = 0;
    uint32_t presentsSinceOff = 0;
    double offSinceMs = 0.0;
};

inline void GateOnWait(GateState& gate, WaitOutcome outcome, double nowMs)
{
    if (outcome == WaitOutcome::Neutral)
        return;

    if (outcome == WaitOutcome::Done)
    {
        gate.timeouts = 0;
        return;
    }

    if (gate.on && ++gate.timeouts >= kGateGiveUp)
    {
        gate.on = false;
        gate.presentsSinceOff = 0;
        gate.offSinceMs = nowMs;
    }
}

// Per game present: a gate that stood aside comes back after a while
inline void GateOnPresent(GateState& gate, double nowMs)
{
    if (gate.on)
        return;

    if (++gate.presentsSinceOff >= kGateRearmPresents || nowMs - gate.offSinceMs >= kGateRearmMs)
    {
        gate.on = true;
        gate.timeouts = 0;
    }
}

// Our own present id counter stays above the ids a game supplies (0 is no id)
inline uint64_t OwnIdAfterGame(uint64_t own, uint64_t gameId) { return gameId > own ? gameId : own; }

// FlipMetering= as f5low reads it (src/oob_driver.h): -1 auto, 0 off ("0", "off", "false", "no"), 1 on ("1", "on",
// "true", "yes"), case and surrounding spaces ignored, anything else counts as auto.
inline int FlipMeteringForced(const char* setting)
{
    char v[8] = {};
    size_t n = 0;

    if (setting == nullptr)
        return -1;

    while (*setting == ' ' || *setting == '\t')
        ++setting;

    for (; *setting != 0; ++setting)
    {
        if (n + 1 >= sizeof(v))
            return -1;

        v[n++] = (*setting >= 'A' && *setting <= 'Z') ? static_cast<char>(*setting - 'A' + 'a') : *setting;
    }

    while (n > 0 && (v[n - 1] == ' ' || v[n - 1] == '\t' || v[n - 1] == '\r' || v[n - 1] == '\n'))
        v[--n] = 0;

    const auto is = [&](const char* word)
    {
        size_t i = 0;
        while (word[i] != 0 && v[i] == word[i])
            ++i;
        return word[i] == 0 && v[i] == 0;
    };

    if (is("0") || is("off") || is("false") || is("no"))
        return 0;

    if (is("1") || is("on") || is("true") || is("yes"))
        return 1;

    return -1;
}

// VK_PRESENT_MODE_FIFO_KHR = 2, FIFO_RELAXED_KHR = 3: V-Sync paces the presents, metering has nothing to add
inline bool VsyncPaces(int presentMode) { return presentMode == 2 || presentMode == 3; }

// V-Sync off on Vulkan: a game that asks for MAILBOX on a screen with a fixed refresh has the driver replace the frames
// above the refresh (a 125 fps picture on 120 Hz loses about 5 frames a second, unevenly), where D3D12 lets f5low pass
// the game's tearing flag through and every frame reaches the screen. So with frame generation on our swapchain is made
// with IMMEDIATE instead (VK_PRESENT_MODE_ IMMEDIATE_KHR = 0, MAILBOX = 1), when the surface offers it, the setting
// allows it and the game's own list of present modes (a VkSwapchainPresentModesCreateInfoEXT, when it chains one) has
// IMMEDIATE in it. FIFO / FIFO_RELAXED and every other mode stay.
enum class TearingWhy : int
{
    NotMailbox,         // the game did not ask for MAILBOX: its mode stays
    FgOff,              // frame generation is off: the game's mode stays
    SettingOff,         // [F5Low] VulkanTearing=false
    NoImmediate,        // the surface does not offer IMMEDIATE
    ListLacksImmediate, // the game chains its own present-mode list or scaling info (compatibility is per mode)
    Tearing             // MAILBOX -> IMMEDIATE
};

struct OurPresentMode
{
    int mode;
    TearingWhy why;
};

inline OurPresentMode ChoosePresentMode(int gameMode, bool immediateSupported, bool listAllowsImmediate,
                                        bool tearingSetting, bool fgOn)
{
    if (gameMode != 1)
        return { gameMode, TearingWhy::NotMailbox };

    if (!fgOn)
        return { gameMode, TearingWhy::FgOff };

    if (!tearingSetting)
        return { gameMode, TearingWhy::SettingOff };

    if (!immediateSupported)
        return { gameMode, TearingWhy::NoImmediate };

    if (!listAllowsImmediate)
        return { gameMode, TearingWhy::ListLacksImmediate };

    return { 0, TearingWhy::Tearing };
}

inline const char* TearingText(TearingWhy why)
{
    switch (why)
    {
    case TearingWhy::NotMailbox:
        return "the game's mode stays";
    case TearingWhy::FgOff:
        return "frame generation is off, the game's mode stays";
    case TearingWhy::SettingOff:
        return "[F5Low] VulkanTearing=false, MAILBOX stays (the driver replaces frames above the refresh rate)";
    case TearingWhy::NoImmediate:
        return "the surface has no IMMEDIATE mode, MAILBOX stays (the driver replaces frames above the refresh rate)";
    case TearingWhy::ListLacksImmediate:
        return "the game chains its own present-mode list or scaling info, MAILBOX stays (the driver replaces frames "
               "above the refresh rate)";
    default:
        return "tearing like D3D12: every generated frame reaches the screen";
    }
}

// The mode of a present the game chained (a VkSwapchainPresentModeInfoEXT): a swapchain made with IMMEDIATE for MAILBOX
// takes IMMEDIATE for the game's MAILBOX presents too
inline int ChainedPresentMode(bool tearing, int chainedMode) { return tearing && chainedMode == 1 ? 0 : chainedMode; }

// Who spaces the frames of a batch, as D3D12 does (f5low pace_sched.h FixedTimer): true = the display (the batch goes
// to the driver back to back), false = our timer. Only a V-Sync / FIFO present can be display paced; MAILBOX /
// IMMEDIATE batches are always timed. With the latency gate on the gate paces by itself: display, whatever the player
// set. Without it f5low's FixedPacing setting decides (0 auto = timer, 1 display, 2 timer). The pacer export takes the
// gate flag and applies the same rule itself (F5LowPacerCreateInfo::gate); this is the host's copy, used by the tests
// to hold the two together and by the log.
inline bool DisplaySpaces(bool gateActive, int presentMode, int fixedPacingSetting)
{
    if (!VsyncPaces(presentMode))
        return false;

    return gateActive || fixedPacingSetting == 1;
}

// The GPU stamps of the frames in flight (one per slot and kind): when the game's frame was finished on the GPU (ready)
// and when f5low's work was (done), QPC. Two kinds, each its own fence and its own watcher: a stamp of one kind never
// waits for the other, whichever finishes first. A stamp belongs to the value it was taken for: a slot used again does
// not hand out the old frame's.
struct StampBook
{
    enum Kind : uint32_t
    {
        Ready = 0,
        Done = 1
    };
    struct One
    {
        uint64_t value = 0;
        int64_t ticks = 0;
    };
    One slot[kSlots][2];

    void Put(Kind kind, uint32_t index, uint64_t value, int64_t ticks)
    {
        slot[index % kSlots][kind] = { value, ticks };
    }

    // 0: not stamped (yet) for this value
    int64_t Take(Kind kind, uint32_t index, uint64_t value) const
    {
        const auto& one = slot[index % kSlots][kind];
        return value != 0 && one.value == value ? one.ticks : 0;
    }
};

// A stamp key: the fence value, with a bit for the path the value came from (the copy-in fence of the shared images or
// the depth stage's own fence count separately and may repeat a value after a path switch)
inline uint64_t StampKey(uint64_t value, bool fromDepthStage)
{
    return fromDepthStage ? (value | (1ull << 63)) : value;
}

// How long a present of the batch waits for f5low's done stamp before telling the pacer: a generated frame of a batch
// with a lead (timed: MAILBOX / IMMEDIATE, or FIFO without the gate) reports how late its own GPU work held it, and the
// pacer lengthens the lead by that. The frame's present waited for the same fence, so the watcher notes it within a
// moment; a batch without a lead takes what is there.
inline uint32_t DoneStampWaitMs(int64_t leadTicks, uint32_t k, uint32_t generated)
{
    return leadTicks > 0 && k >= 1 && k <= generated ? 2u : 0u;
}

// The pacer's own frame interval, from the batch it planned (step = interval / (generated + 1)); the CPU-only game time
// is not it
inline double PacerIntervalMs(int64_t stepTicks, uint32_t generated, int64_t ticksPerSecond)
{
    if (stepTicks <= 0 || ticksPerSecond <= 0)
        return 0.0;

    return static_cast<double>(stepTicks) * static_cast<double>(generated + 1) * 1000.0 /
           static_cast<double>(ticksPerSecond);
}

// Hardware flip metering (VK_NV_present_metering) without the pacer export (an older f5low.dll): off unless forced on
// (FlipMetering=1), and never on a V-Sync swapchain or without the extension on the device. The pacer export decides
// it itself when there is one.
inline bool MeteringOn(int forced, bool deviceHasIt, bool vsyncPaces)
{
    if (!deviceHasIt || vsyncPaces)
        return false;

    return forced == 1;
}

// The driver's feedback on a metered batch: anything reported counts against it until its meaning is known, and after
// kMeterRefusals in a row it is off for the session
inline constexpr uint32_t kMeterRefusals = 8;

struct MeterHealth
{
    uint32_t bad = 0;
    bool off = false;
};

inline void MeterNote(MeterHealth& health, bool negative)
{
    if (health.off)
        return;

    if (!negative)
        health.bad = 0;
    else if (++health.bad >= kMeterRefusals)
        health.off = true;
}

inline bool MeterNegative(bool presentFailed, uint32_t feedback) { return presentFailed || feedback != 0; }

// numFramesPerBatch on a present: a metered batch sends its presented frames + 1 (a cut-short batch is smaller), the
// first present after one that was not metered sends 1 (the batch is closed), otherwise nothing is chained (0).
inline uint32_t BatchConfig(bool metered, bool lastWasMetered, uint32_t presentedGenerated)
{
    if (metered)
        return presentedGenerated + 1;

    return lastWasMetered ? 1u : 0u;
}

// The generated frames of a batch are spaced on the CPU unless the driver meters them (the first present of a
// metered batch still goes out on its schedule)
inline bool PacedOnCpu(bool metered) { return !metered; }

// Which swapchains generate (F5Low on D3D12 takes 8-bit RGBA/BGRA and HDR10; FP16/scRGB are refused). VkFormat numbers:
// R8G8B8A8 UNORM 37 / SRGB 43, B8G8R8A8 UNORM 44 / SRGB 50, A2R10G10B10_UNORM_PACK32 58, A2B10G10R10_UNORM_PACK32 64,
// R16G16B16A16_SFLOAT 97; colour space HDR10_ST2084_EXT 1000104008. A2B10G10R10 is DXGI's R10G10B10A2; A2R10G10B10 has
// no DXGI twin (channel order), so it is refused.
enum class SwapchainSupport
{
    Ok,
    OkHdr10,
    RefusedFp16,
    RefusedOrder,
    RefusedColourSpace,
    RefusedFormat
};

inline SwapchainSupport ClassifySwapchain(int format, long long colorSpace)
{
    switch (format)
    {
    case 37:
    case 43:
    case 44:
    case 50:
        return SwapchainSupport::Ok;
    case 64:
        return colorSpace == 1000104008 ? SwapchainSupport::OkHdr10 : SwapchainSupport::RefusedColourSpace;
    case 58:
        return SwapchainSupport::RefusedOrder;
    case 97:
        return SwapchainSupport::RefusedFp16;
    default:
        return SwapchainSupport::RefusedFormat;
    }
}

inline bool SwapchainGenerates(SwapchainSupport support)
{
    return support == SwapchainSupport::Ok || support == SwapchainSupport::OkHdr10;
}

// The reasons, in the words of the D3D12 F5Low checks where there is one
inline const char* SwapchainReason(SwapchainSupport support)
{
    switch (support)
    {
    case SwapchainSupport::RefusedFp16:
        return "only HDR10 output is supported";
    case SwapchainSupport::RefusedOrder:
        return "the swapchain's 10-bit format has its channels in an order the generator does not take (A2R10G10B10), "
               "only A2B10G10R10 is supported";
    case SwapchainSupport::RefusedColourSpace:
        return "a 10-bit swapchain only works with the HDR10 (ST 2084) colour space";
    case SwapchainSupport::RefusedFormat:
        return "the swapchain's format is not supported (8-bit RGBA/BGRA and HDR10 are)";
    default:
        return "";
    }
}

// What a frame's hudless picture does for the generator (the D3D12 uiMode: the hudless one when the input has it and it
// matches, else colour only)
enum class HudlessMode
{
    ColourOnly, // the input offers none
    Used,
    NoLayout,     // offered, but not in a readable state
    SizeMismatch, // not the swapchain's size
    FormatMismatch
};

inline HudlessMode HudlessModeOf(bool offered, bool layoutOk, bool sizeOk, bool formatOk)
{
    if (!offered)
        return HudlessMode::ColourOnly;

    if (!layoutOk)
        return HudlessMode::NoLayout;

    if (!sizeOk)
        return HudlessMode::SizeMismatch;

    return formatOk ? HudlessMode::Used : HudlessMode::FormatMismatch;
}

inline const char* HudlessText(HudlessMode mode)
{
    switch (mode)
    {
    case HudlessMode::Used:
        return "the game's hudless picture is passed to the generator";
    case HudlessMode::NoLayout:
        return "colour only (the hudless picture is not in a readable state)";
    case HudlessMode::SizeMismatch:
        return "colour only (the hudless picture is not the swapchain's size)";
    case HudlessMode::FormatMismatch:
        return "colour only (the hudless picture's format differs from the swapchain's)";
    default:
        return "colour only (the input offers no hudless picture)";
    }
}

// Dynamic MFG on Vulkan: the setting is the most; it runs unless the count follows the game's DLSS FG. A count of 0 is
// 1X (generate nothing, still feed the generator); the count kept for when it is back is `kept`.
inline bool DynamicRuns(bool dynamicSet, bool followsGame, int maxCount)
{
    return dynamicSet && !followsGame && maxCount > 1;
}

// Frames asked of the generator: 0 while Dynamic MFG holds at 1X (history is still fed)
inline uint32_t CountToGenerate(bool dynamicHold, uint32_t count) { return dynamicHold ? 0u : count; }

inline int CountAfterDynamic(int chosen, int kept) { return chosen >= 1 ? chosen : kept; }

// Whether "Dynamic MFG holds back" is true: it chose 1X (the status line says so only then)
inline bool DynamicHolds(bool running, int chosen) { return running && chosen == 0; }

// Whether the game's window is the foreground one (the count is kept while it is not): its process owns it
inline bool InFront(unsigned long foregroundPid, unsigned long ownPid)
{
    return foregroundPid != 0 && foregroundPid == ownPid;
}

// Whether the game's own Reflex cap is divided for F5Low's shown frames: per rendered frame when it generates, no
// driver multiplier on Vulkan (the driver never hears our multiplier; its minimum interval counts the game's own
// frames)
inline float ReflexCapForVulkanF5Low(float shownCapFps, uint32_t shownPerRendered)
{
    return shownPerRendered > 1 ? shownCapFps / static_cast<float>(shownPerRendered) : shownCapFps;
}

// The pacer export of f5low.dll (f5low_pacer.h): used when every f5lowPacer* function was found and the pacer was made;
// otherwise the spacing is the plain one.
inline bool UsePacer(bool exportsFound, bool created) { return exportsFound && created; }

// F5LOW_PACER_FLAG_HOST_CAN_METER (0x4) only with VK_NV_present_metering and its feature on the device; the burst flag
// (0x2) is never set (no swapchain made for it).
inline uint32_t PacerFlags(bool meteringOnDevice) { return meteringOnDevice ? 0x4u : 0u; }

// Whether this batch is metered: the pacer's answer (meterWanted) when there is one, else the fallback rule; always
// needs the extension on the device
inline bool MeterThisBatch(bool pacerPresent, uint32_t meterWanted, bool fallbackOn, bool meteringUsable)
{
    return meteringUsable && (pacerPresent ? meterWanted != 0 : fallbackOn);
}

// The status handed to f5lowPacerMeterResult: 0 accepted, 1 a refusal (a failed present or presentConfigFeedback != 0)
inline int32_t MeterStatus(bool presentFailed, uint32_t feedback) { return (presentFailed || feedback != 0) ? 1 : 0; }

// The 1 ms timer follows generating, with hysteresis: on at once, off after kTimerOffFrames frames without generating
inline constexpr uint32_t kTimerOffFrames = 30;

struct TimerState
{
    bool raised = false;
    uint32_t idle = 0;
};

// True when the raise should change (to `state.raised` after the call)
inline bool TimerUpdate(TimerState& state, bool generating)
{
    if (generating)
    {
        state.idle = 0;

        if (state.raised)
            return false;

        state.raised = true;
        return true;
    }

    if (!state.raised)
        return false;

    if (++state.idle < kTimerOffFrames)
        return false;

    state.raised = false;
    state.idle = 0;
    return true;
}

// The host says it can meter only when the ini asks for it, the device has VK_NV_present_metering and it was not
// refused
inline bool HostCanMeter(bool iniOn, bool onDevice) { return iniOn && onDevice; }

// A pacer wait is clamped on our side: never past two intervals or 100 ms from now
inline int64_t ClampRelease(int64_t releaseTicks, int64_t nowTicks, int64_t ticksPerSecond, double intervalMs)
{
    const int64_t hundredMs = ticksPerSecond / 10;
    const int64_t twoIntervals =
        intervalMs > 0.0 ? static_cast<int64_t>(2.0 * intervalMs * ticksPerSecond / 1000.0) : hundredMs;
    const int64_t limit = nowTicks + (twoIntervals < hundredMs ? twoIntervals : hundredMs);
    return releaseTicks < limit ? releaseTicks : limit;
}

// Present spacing of the shown frames (ms): mean, standard deviation, largest, and how many ran late
struct SpacingStats
{
    uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double max = 0.0;
    uint64_t late = 0;

    void Add(double ms)
    {
        ++count;
        const double delta = ms - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (ms - mean);

        if (ms > max)
            max = ms;
    }

    double Sd() const { return count > 1 ? std::sqrt(m2 / static_cast<double>(count - 1)) : 0.0; }
};

// A present is late when it came more than half a step after its slot
inline bool Late(double presentMs, double slotMs, double stepMs) { return presentMs > slotMs + stepMs * 0.5; }

// The stats line comes every this many game frames
inline constexpr uint32_t kStatsEvery = 120;

// Whether the frame may be generated at all: the interval is known and the generator has a count to make.
inline bool CanPace(double intervalMs, uint32_t count) { return intervalMs > 0.0 && count > 0; }
} // namespace F5LowVk
