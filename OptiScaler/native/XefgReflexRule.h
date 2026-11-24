#pragma once

// A game's own Reflex while XeFG is the frame generation output: InputCommon (low_latency/input, OutputRule.h) runs
// XeFG's XeLL as the output, or next to Reflex / Anti-Lag 2 with its low latency off, for a D3D game and for a Vulkan
// game on the present bridge (native/VkPresentBridge.h, XeLL on the bridge's private D3D12 device). These are the sleep
// mode and frame cap rules for both. No Windows, no Vulkan, no GPU: a host test can include it.

#include "NativeLowLatencyRule.h"

#include <cstdint>

namespace native::xefgreflex
{
// XeFG runs on the Vulkan present bridge (its XeLL on the bridge's D3D12 device): XeFG has to be the output and the
// bridge's device has to be up. Never for a game on dxvk without the Vulkan bridge: it has no bridge, the D3D11 side
// runs XeFG.
inline bool ShouldForward(bool xefgOutput, bool bridgeDevicePresent, bool dxvkWithoutVulkanBridge)
{
    return xefgOutput && bridgeDevicePresent && !dxvkWithoutVulkanBridge;
}

struct SleepMode
{
    bool lowLatency = false;
    bool boost = false;
    uint32_t minimumIntervalUs = 0; // 0: no limit
    bool useMarkersToOptimize = false;
};

// The game turned its own Reflex off: its Reflex calls are the input, it has called SetSleepMode, and that mode has low
// latency off. A game that only marks frames or sleeps without ever setting a mode, OptiScaler's own calls and the UE
// mod's never count: nothing was turned off there. Not while a frame generation needs Reflex on
// (native::lowlatency::FrameGenerationNeedsReflex), nor with fakenvapi's Force Enable (XeLL's sleeps run then).
// InputCommon::game_turned_low_latency_off is the one place that asks: XeLL's sleep mode, the frame cap and the menu
// must agree, or the cap can end up with no owner.
constexpr bool GameTurnedLowLatencyOff(bool gameReflexIsInput, bool gameSetMode, bool gameLowLatencyOn,
                                       bool frameGenerationNeedsReflex, bool forceEnabled = false)
{
    return gameReflexIsInput && gameSetMode && !gameLowLatencyOn && !frameGenerationNeedsReflex && !forceEnabled;
}

// The sleep mode the game's Reflex output gets. OptiScaler's frame cap replaces the game's interval when one is set.
// While XeFG is the output (its XeLL, D3D or the bridge), low latency is on: XeFG always uses XeLL, and an input that
// never said anything about low latency (a game that only marks frames) would leave it doing nothing. It goes off when
// the game turned its own Reflex off (`gameTurnedOff`, GameTurnedLowLatencyOff: the game's setting is followed, as
// upstream does) or with fakenvapi's Force Disable. XeLL then skips every sleep and marker, so its frame cap cannot
// wait there either: OptiScaler's limiter keeps the cap (the caller hands XellHoldsCap the same answer; Reflex off in
// RDR2's menu once left the cap doing nothing). Not written for XeLL alone: whichever low latency XeFG is paired with
// gets the same.
// `blockLowLatency` (the Framerate page's Input: Off, any output): low latency off, no Boost, no interval whatever the
// game or the cap says; OptiScaler's limiter keeps the cap (XellHoldsCap and ReflexHooks::update do not hand it over).
inline SleepMode SleepModeForOutput(const SleepMode& game, uint32_t capIntervalUs, bool xefgOutput, bool forceDisabled,
                                    bool blockLowLatency = false, bool gameTurnedOff = false)
{
    SleepMode out = game;

    if (blockLowLatency)
    {
        out.lowLatency = false;
        out.boost = false;
        out.minimumIntervalUs = 0;
        return out;
    }

    if (capIntervalUs != 0)
        out.minimumIntervalUs = capIntervalUs;

    if (xefgOutput)
        out.lowLatency = !forceDisabled && !gameTurnedOff;

    return out;
}

// ---- XeFG with Reflex or Anti-Lag 2 as the low latency output ("split") ----
//
// XeFG needs an XeLL context and its frame IDs follow the markers sent (framegen/xefg/XeFGFrameId.h), so XeLL stays in
// place and every call still reaches it, but with its low latency off: XeLL skips every sleep and marker while its mode
// is off (as with Force Disable), and XeFG keeps running. The latency work is the other side's: InputCommon's output,
// the driver's Reflex or Anti-Lag 2 / Anti-Lag Vk, gets the game's real mode, markers and Sleep, the way FSR FG's
// Reflex does.

using lowlatency::XefgLatency;

// The side that lowers latency right now. XeLL is today's behaviour and the answer whenever the setup lacks the choice,
// XeFG is not the output, the Framerate page's Input is Off (nothing to lower) or fakenvapi's Force Reflex is set (Force
// Disable wants everything off; Force Enable would make XeLL sleep next to the other side).
inline XefgLatency EffectiveXefgLatency(const lowlatency::OutputInputs& in, bool blockLowLatency, bool forceReflexSet)
{
    if (!in.xefgOutput || blockLowLatency || forceReflexSet || !lowlatency::XefgLatencyAvailable(in.xefgLatency, in))
        return XefgLatency::XeLL;

    return in.xefgLatency;
}

// The game's sleep mode as XeLL gets it: low latency off, no Boost, no interval (it neither sleeps nor holds the cap)
inline SleepMode XellSleepModeInSplit(const SleepMode& game)
{
    SleepMode out = game;
    out.lowLatency = false;
    out.boost = false;
    out.minimumIntervalUs = 0;
    return out;
}

// ...and as the other side gets it: the game's own, the cap in place of its interval when one is set
inline SleepMode DriverSleepModeInSplit(const SleepMode& game, uint32_t capIntervalUs)
{
    SleepMode out = game;

    if (capIntervalUs != 0)
        out.minimumIntervalUs = capIntervalUs;

    return out;
}

// Whether XeFG's XeLL keeps the frame cap the game's Reflex hands it: while it is not idle (NativeLowLatencyRule.h
// XellIdle: on NVIDIA and AMD it does nothing while XeFG generates no frames), not turned off (`forceDisabled`:
// fakenvapi's Force Disable, Input Off, the split, or the game turned its own Reflex off, GameTurnedLowLatencyOff) and
// frame generation presents without V-Sync (XellKeepsCap). OptiScaler's limiter keeps it otherwise.
inline bool XellHoldsCap(bool forceDisabled, bool intelGpu, bool xefgGenerating, bool xefgVsync)
{
    // (forceDisabled: any reason XeLL's sleeps do not run, XellSleepsOff)
    lowlatency::CapInputs in;
    in.method = lowlatency::Method::XellWithXefg;
    in.intelGpu = intelGpu;
    in.xefgGenerating = xefgGenerating;
    return !forceDisabled && !lowlatency::XellIdle(in) && lowlatency::XellKeepsCap(xefgVsync);
}

// Why XeFG's XeLL does not sleep, so cannot keep the cap (XellHoldsCap's first argument): Force Disable, Input Off,
// another output in the split, or the game turned its own Reflex off (the same answer SleepModeForOutput gets)
constexpr bool XellSleepsOff(bool forceDisabled, bool blocked, bool gameTurnedOff, bool split)
{
    return forceDisabled || blocked || gameTurnedOff || split;
}

} // namespace native::xefgreflex
