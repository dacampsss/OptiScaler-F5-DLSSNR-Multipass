// Host check of F5Low on a native Vulkan swapchain: F5LowVkRules.h (image count, headroom, generated count, pacing).
// cl /std:c++20 /EHsc /W4 tests/f5low_vk_rules_smoke.cpp
#include "../OptiScaler/framegen/f5low/F5LowVkRules.h"

#include <cmath>
#include <string>
#include <cstdio>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                                                           \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

int main()
{
    using namespace F5LowVk;

    // image count: game's plus the generated ones, capped by the surface
    CHECK(RequestedMinImageCount(3, 5, 0) == 8);
    CHECK(RequestedMinImageCount(3, 5, 8) == 8);
    CHECK(RequestedMinImageCount(3, 5, 6) == 6);
    CHECK(RequestedMinImageCount(3, 5, 2) == 3); // never below the game's own request
    CHECK(RequestedMinImageCount(2, 0, 0) == 2);

    // headroom
    CHECK(ImageHeadroom(8, 3) == 5);
    CHECK(ImageHeadroom(3, 3) == 0);
    CHECK(ImageHeadroom(2, 3) == 0);

    // generated count
    CHECK(GeneratedCount(3, 5, 5) == 3);
    CHECK(GeneratedCount(5, 3, 5) == 3);  // generator limit
    CHECK(GeneratedCount(5, 5, 2) == 2);  // image headroom
    CHECK(GeneratedCount(9, 9, 9) == 5);  // API limit
    CHECK(GeneratedCount(0, 5, 5) == 0);
    CHECK(GeneratedCount(-1, 5, 5) == 0);
    CHECK(GeneratedCount(3, 5, 0) == 0);

    // interval smoothing
    CHECK(NextInterval(0.0, 16.6) == 16.6);
    CHECK(NextInterval(0.0, 500.0) == 0.0);
    CHECK(NextInterval(16.0, 500.0) == 16.0);
    CHECK(NextInterval(16.0, 0.5) == 16.0);
    CHECK(std::fabs(NextInterval(20.0, 10.0) - 18.0) < 1e-9);

    // spacing: waits stay below one interval
    CHECK(SpacingMs(0.0, 3) == 0.0);
    CHECK(SpacingMs(20.0, 0) == 0.0);
    CHECK(std::fabs(SpacingMs(20.0, 3) - 5.0) < 1e-9);
    for (uint32_t count = 1; count <= kMaxGenerated; ++count)
        CHECK(SpacingMs(33.3, count) * count < 33.3);

    CHECK(CanPace(16.6, 2));
    CHECK(!CanPace(0.0, 2));
    CHECK(!CanPace(16.6, 0));

    // fences: never wait on a value that will not arrive
    CHECK(!FenceBehind(5, 5));
    CHECK(!FenceBehind(4, 5));
    CHECK(FenceBehind(3, 5));
    CHECK(FenceBehind(UINT64_MAX, 5));
    CHECK(!FenceBehind(0, 0));

    // fake fence model: NOT_READY, an error and OK frames; a failed submit takes no value, every taken value is higher
    {
        FenceCounter ready, consumed, done;
        uint64_t lastConsumed = 0;
        const int results[] = { 1 /*NOT_READY*/, -5 /*error*/, 0, 0, 1, 0 };
        const uint32_t reported[] = { 0, 0, 0, 3, 0, 3 };
        const bool submitOk[] = { true, true, false, true, true, true };
        uint32_t shown = 0;

        for (int i = 0; i < 6; ++i)
        {
            const uint64_t copied = ready.Peek();
            CHECK(copied == ready.issued + 1);

            if (!submitOk[i])
                continue; // nothing taken, f5low not called

            ready.Take();
            const uint64_t c = consumed.Take(), d = done.Take(); // f5low signals both for every answer
            CHECK(c > lastConsumed);
            lastConsumed = c;
            CHECK(d == c);
            shown += GeneratedAfter(results[i], reported[i], 3);
        }

        CHECK(ready.issued == 5);
        CHECK(consumed.issued == 5);
        CHECK(shown == 6); // the two OK frames with all 3
    }

    CHECK(GeneratedAfter(0, 3, 3) == 3);
    CHECK(GeneratedAfter(0, 0, 3) == 0);
    CHECK(GeneratedAfter(1, 3, 3) == 0);
    CHECK(GeneratedAfter(-2, 3, 3) == 0);

    // asking the game for a new swapchain after a raise: suboptimal, then out of date once, then give up
    {
        RecreateAsk ask;
        CHECK(!OnPresentRecreate(ask, 1, 1, 1, 0.0));  // nothing raised
        CHECK(OnPresentRecreate(ask, 3, 1, 1, 10.0));  // raised: the suboptimal ask
        CHECK(ask.state == RecreateState::Asked);
        CHECK(!OnPresentRecreate(ask, 3, 1, 1, 20.0)); // not again
        for (uint32_t i = 0; i < kAskPresents + 2; ++i)
            CHECK(!OnPresentRecreate(ask, 3, 1, 1, 100.0)); // 30 presents but not 500 ms yet
        CHECK(ask.state == RecreateState::Asked);
        CHECK(!OnAcquireRecreate(ask, 120.0)); // not due
        CHECK(!OnPresentRecreate(ask, 3, 1, 1, 600.0));
        CHECK(ask.state == RecreateState::ForcePending);
        CHECK(OnAcquireRecreate(ask, 610.0)); // out of date, once
        CHECK(!OnAcquireRecreate(ask, 611.0));
        CHECK(ask.state == RecreateState::Forced);
        CHECK(!OnPresentRecreate(ask, 3, 1, 1, 2000.0));
        CHECK(ask.state == RecreateState::Forced);
        CHECK(!OnPresentRecreate(ask, 3, 1, 1, 2700.0)); // 2 s without a new swapchain
        CHECK(ask.state == RecreateState::GaveUp);
        CHECK(!OnPresentRecreate(ask, 3, 1, 1, 9000.0)); // stays given up for this count
        CHECK(!OnAcquireRecreate(ask, 9001.0));
        CHECK(OnPresentRecreate(ask, 4, 1, 1, 9100.0)); // another count starts over
        CHECK(ask.state == RecreateState::Asked);
        CHECK(!OnPresentRecreate(ask, 2, 1, 2, 9200.0)); // lowered: idle
        CHECK(ask.state == RecreateState::Idle);
        CHECK(NeedsRaise(3, 1, 1));
        CHECK(!NeedsRaise(3, 3, 1));
        CHECK(!NeedsRaise(3, 1, 3));
    }

    // device group present info: one device only
    CHECK(SingleDeviceMask(1));
    CHECK(SingleDeviceMask(4));
    CHECK(!SingleDeviceMask(0));
    CHECK(!SingleDeviceMask(3));
    CHECK(DeviceGroupCarryable(1, 1, 1));
    CHECK(DeviceGroupCarryable(0, 1, 0));
    CHECK(!DeviceGroupCarryable(1, 1, 3)); // two devices
    CHECK(!DeviceGroupCarryable(1, 4, 1)); // SUM mode
    CHECK(!DeviceGroupCarryable(2, 1, 1)); // two swapchains

    // spare images up front, the latency bound and its gate
    CHECK(SpareImagesUpFront(5) == 5);
    CHECK(SpareImagesUpFront(3) == 3);
    CHECK(SpareImagesUpFront(0) == 1);
    CHECK(SpareImagesUpFront(9) == 5);
    CHECK(SpareImagesToAsk(true, 5, 2) == 5);  // bounded: the most
    CHECK(SpareImagesToAsk(false, 5, 2) == 2); // no bound: what the multiplier needs
    CHECK(SpareImagesToAsk(false, 3, 5) == 3);
    CHECK(LatencyBound(4, 3, true) == 3);
    CHECK(LatencyBound(3, 3, true) == 2);
    CHECK(LatencyBound(2, 3, true) == 2); // never below 2
    CHECK(LatencyBound(1, 3, true) == 2);
    CHECK(LatencyBound(3, 1, true) == 3); // 2X: a batch of two images needs the game one batch ahead
    CHECK(LatencyBound(2, 1, true) == 3);
    CHECK(LatencyBound(5, 1, true) == 4); // the game's own bound when it is higher
    CHECK(LatencyBound(3, 2, true) == 2 && LatencyBound(3, 5, true) == 2);
    CHECK(LatencyBound(3, 5, false) == 3 && LatencyBound(6, 4, false) == 5); // a timed batch: a whole batch ahead
    CHECK(GateUsable(3));
    CHECK(!GateUsable(kGateRing));
    CHECK(GateWaitNumber(0, 3) == 0); // nothing issued: no wait
    CHECK(GateWaitNumber(2, 3) == 0); // third present may go out
    CHECK(GateWaitNumber(3, 3) == 1); // the fourth needs the first on screen
    CHECK(GateWaitNumber(10, 3) == 8);
    CHECK(OwnIdAfterGame(5, 9) == 9);
    CHECK(OwnIdAfterGame(9, 5) == 9);

    // the gate gives up after 5 timeouts in a row, neutral results change nothing, it comes back later
    {
        GateState gate;
        for (uint32_t i = 0; i < kGateGiveUp - 1; ++i)
            GateOnWait(gate, WaitOutcome::Timeout, 10.0);
        CHECK(gate.on);
        GateOnWait(gate, WaitOutcome::Neutral, 11.0);
        CHECK(gate.on && gate.timeouts == kGateGiveUp - 1);
        GateOnWait(gate, WaitOutcome::Done, 12.0); // a good wait clears the count
        CHECK(gate.timeouts == 0);
        for (uint32_t i = 0; i < kGateGiveUp; ++i)
            GateOnWait(gate, WaitOutcome::Timeout, 100.0);
        CHECK(!gate.on);
        GateOnPresent(gate, 200.0);
        CHECK(!gate.on);
        for (uint32_t i = 0; i < kGateRearmPresents; ++i)
            GateOnPresent(gate, 200.0 + i);
        CHECK(gate.on && gate.timeouts == 0); // 120 presents
        for (uint32_t i = 0; i < kGateGiveUp; ++i)
            GateOnWait(gate, WaitOutcome::Timeout, 1000.0);
        CHECK(!gate.on);
        GateOnPresent(gate, 1000.0 + kGateRearmMs); // or 3 s
        CHECK(gate.on);
    }

    // flip metering: f5low's words; without the pacer export it is off unless forced on, never on V-Sync or without
    // the device extension
    CHECK(FlipMeteringForced("auto") == -1);
    CHECK(FlipMeteringForced("") == -1);
    CHECK(FlipMeteringForced(nullptr) == -1);
    CHECK(FlipMeteringForced("0") == 0);
    CHECK(FlipMeteringForced(" Off ") == 0);
    CHECK(FlipMeteringForced("FALSE") == 0);
    CHECK(FlipMeteringForced("no") == 0);
    CHECK(FlipMeteringForced("1") == 1);
    CHECK(FlipMeteringForced("On") == 1);
    CHECK(FlipMeteringForced("yes") == 1);
    CHECK(FlipMeteringForced("sometimes") == -1);
    CHECK(FlipMeteringForced("averyverylongword") == -1);
    CHECK(!MeteringOn(-1, true, false)); // auto = off here
    CHECK(MeteringOn(1, true, false));   // forced on
    CHECK(!MeteringOn(1, true, true));   // V-Sync paces
    CHECK(!MeteringOn(1, false, false)); // no extension
    CHECK(!MeteringOn(0, true, false));
    CHECK(PacedOnCpu(false) && !PacedOnCpu(true));
    CHECK(VsyncPaces(2) && VsyncPaces(3) && !VsyncPaces(0) && !VsyncPaces(1));

    // the driver's feedback: 8 bad in a row turn it off, a good one clears the count
    {
        MeterHealth health;
        for (uint32_t i = 0; i < kMeterRefusals - 1; ++i)
            MeterNote(health, MeterNegative(false, 1));
        CHECK(!health.off);
        MeterNote(health, MeterNegative(false, 0));
        CHECK(health.bad == 0);
        for (uint32_t i = 0; i < kMeterRefusals; ++i)
            MeterNote(health, MeterNegative(true, 0));
        CHECK(health.off);
        MeterNote(health, false);
        CHECK(health.off);
    }

    // batch size on a present: presented + 1 when metered, 1 to close a metered run, else none
    CHECK(BatchConfig(true, false, 3) == 4);
    CHECK(BatchConfig(true, true, 1) == 2);  // a cut-short batch
    CHECK(BatchConfig(true, true, 0) == 1);
    CHECK(BatchConfig(false, true, 0) == 1); // closes the metered run
    CHECK(BatchConfig(false, false, 0) == 0);

    // swapchain formats: 8-bit and HDR10 generate, FP16 / scRGB and the A2R10G10B10 order are refused with a reason
    CHECK(ClassifySwapchain(44, 0) == SwapchainSupport::Ok);
    CHECK(ClassifySwapchain(50, 0) == SwapchainSupport::Ok);
    CHECK(ClassifySwapchain(37, 0) == SwapchainSupport::Ok);
    CHECK(ClassifySwapchain(64, 1000104008LL) == SwapchainSupport::OkHdr10);
    CHECK(ClassifySwapchain(64, 0) == SwapchainSupport::RefusedColourSpace); // 10-bit, not HDR10: refused
    CHECK(ClassifySwapchain(58, 1000104008LL) == SwapchainSupport::RefusedOrder);
    CHECK(ClassifySwapchain(97, 1000104002LL) == SwapchainSupport::RefusedFp16);
    CHECK(ClassifySwapchain(97, 0) == SwapchainSupport::RefusedFp16);
    CHECK(ClassifySwapchain(123, 0) == SwapchainSupport::RefusedFormat);
    CHECK(SwapchainGenerates(SwapchainSupport::OkHdr10) && !SwapchainGenerates(SwapchainSupport::RefusedFp16));
    CHECK(std::string(SwapchainReason(SwapchainSupport::RefusedFp16)) == "only HDR10 output is supported");
    CHECK(std::string(SwapchainReason(SwapchainSupport::Ok)).empty());

    // hudless
    CHECK(HudlessModeOf(false, true, true, true) == HudlessMode::ColourOnly);
    CHECK(HudlessModeOf(true, false, true, true) == HudlessMode::NoLayout);
    CHECK(HudlessModeOf(true, true, false, true) == HudlessMode::SizeMismatch);
    CHECK(HudlessModeOf(true, true, true, false) == HudlessMode::FormatMismatch);
    CHECK(HudlessModeOf(true, true, true, true) == HudlessMode::Used);
    CHECK(std::string(HudlessText(HudlessMode::Used)).find("hudless") != std::string::npos);

    // Dynamic MFG glue
    CHECK(DynamicRuns(true, false, 3));
    CHECK(!DynamicRuns(true, true, 3));  // follows the game
    CHECK(!DynamicRuns(false, false, 3));
    CHECK(!DynamicRuns(true, false, 1)); // nothing to choose from
    CHECK(CountAfterDynamic(0, 3) == 3); // 1X: the kept count is what comes back
    CHECK(CountAfterDynamic(2, 3) == 2);
    CHECK(DynamicHolds(true, 0));
    CHECK(!DynamicHolds(true, 2));
    CHECK(!DynamicHolds(false, 0));
    CHECK(InFront(100, 100) && !InFront(100, 200) && !InFront(0, 0));

    // the game's Reflex cap for F5Low on Vulkan: per rendered frame, no driver multiplier
    CHECK(std::fabs(ReflexCapForVulkanF5Low(120.0f, 3) - 40.0f) < 1e-4f);
    CHECK(ReflexCapForVulkanF5Low(120.0f, 1) == 120.0f);

    // pacer glue: the export or the plain spacing, flags, metering, status
    CHECK(UsePacer(true, true) && !UsePacer(true, false) && !UsePacer(false, true));
    CHECK(PacerFlags(true) == 0x4u && PacerFlags(false) == 0u);
    CHECK((PacerFlags(true) & 0x2u) == 0); // never the burst
    CHECK(MeterThisBatch(true, 1, false, true));   // the pacer wants it
    CHECK(!MeterThisBatch(true, 0, true, true));   // the pacer does not, whatever the fallback says
    CHECK(MeterThisBatch(false, 0, true, true));   // no pacer: the fallback rule
    CHECK(!MeterThisBatch(false, 0, false, true));
    CHECK(!MeterThisBatch(true, 1, false, false)); // no extension on the device
    CHECK(MeterStatus(false, 0) == 0);
    CHECK(MeterStatus(true, 0) == 1);
    CHECK(MeterStatus(false, 4) == 1);
    {
        TimerState timer;
        CHECK(TimerUpdate(timer, true) && timer.raised);
        CHECK(!TimerUpdate(timer, true));
        for (uint32_t i = 0; i + 1 < kTimerOffFrames; ++i)
            CHECK(!TimerUpdate(timer, false));
        CHECK(!TimerUpdate(timer, true)); // back before it dropped: stays on, count restarts
        for (uint32_t i = 0; i + 1 < kTimerOffFrames; ++i)
            CHECK(!TimerUpdate(timer, false));
        CHECK(TimerUpdate(timer, false) && !timer.raised);
    }
    CHECK(HostCanMeter(true, true) && !HostCanMeter(false, true) && !HostCanMeter(true, false));
    CHECK(ClampRelease(1000, 0, 1000, 10.0) == 20);          // two intervals (10 ms = 10 ticks at 1 kHz) from now
    CHECK(ClampRelease(5, 0, 1000, 10.0) == 5);              // earlier than the limit: kept
    CHECK(ClampRelease(100000, 0, 1000, 500.0) == 100);      // 100 ms cap
    CHECK(CountToGenerate(true, 3) == 0 && CountToGenerate(false, 3) == 3);
    {
        SpacingStats stats;
        stats.Add(8.0);
        stats.Add(10.0);
        stats.Add(12.0);
        CHECK(std::fabs(stats.mean - 10.0) < 1e-9);
        CHECK(std::fabs(stats.Sd() - 2.0) < 1e-9);
        CHECK(stats.max == 12.0);
        CHECK(Late(21.0, 10.0, 8.0) && !Late(13.0, 10.0, 8.0));
    }

    // no runaway: a 60 fps game (16.7 ms of its own) whose thread is held by our waits keeps its interval
    for (int n = 1; n <= 3; ++n)
    {
        double interval = 0.0, lastEntry = 0.0, now = 0.0, wait = 0.0;
        const double ownMs = 16.7;

        for (int frame = 0; frame < 600; ++frame)
        {
            now += ownMs + wait; // the game's own time plus what our waits held it

            if (lastEntry > 0.0)
                interval = NextInterval(interval, Unthrottled(now - lastEntry, wait));

            lastEntry = now;

            // Our waits next frame: 3 ms of gate/drain, plus any CPU spacing (none under FIFO)
            wait = 3.0 + BatchSpacingMs(interval, static_cast<uint32_t>(n), true) * n;
        }

        CHECK(std::fabs(interval - 16.7) < 0.2);
        CHECK(std::min(120.0, 60.0 * (n + 1)) == 120.0); // 120 Hz FIFO shows min(refresh, 60 * (n + 1))
        CHECK(BatchSpacingMs(16.7, static_cast<uint32_t>(n), false) * (n + 1) <= 16.7 - 0.5 + 1e-9);
        CHECK(BatchSpacingMs(16.7, static_cast<uint32_t>(n), true) == 0.0);
    }
    CHECK(Unthrottled(20.0, 5.0) == 15.0 && Unthrottled(10.0, 30.0) < 0.0);

    // who spaces a batch: only V-Sync / FIFO can be display paced; the gate makes it so, else the player's setting (1 = display)
    for (int mode = 0; mode <= 3; ++mode)
        for (int setting = 0; setting <= 2; ++setting)
        {
            CHECK(DisplaySpaces(false, mode, setting) == (VsyncPaces(mode) && setting == 1));
            CHECK(DisplaySpaces(true, mode, setting) == VsyncPaces(mode));
        }
    CHECK(!DisplaySpaces(true, 0, 0) && !DisplaySpaces(true, 1, 2)); // IMMEDIATE / MAILBOX are timed with the gate too
    CHECK(DisplaySpaces(true, 2, 2) && DisplaySpaces(true, 3, 0));   // the gate over a "timer" setting

    // the GPU stamps: a kind never waits for the other, a slot used again hands out nothing of the old frame
    {
        StampBook book;
        CHECK(book.Take(StampBook::Ready, 0, 10) == 0 && book.Take(StampBook::Done, 0, 10) == 0);
        // frame N = 10 and N+1 = 11: ready 10 and ready 11 are stamped before f5low's done 10 (which finishes after ready 11)
        book.Put(StampBook::Ready, 0, 10, 1000);
        book.Put(StampBook::Ready, 1, 11, 2000);
        CHECK(book.Take(StampBook::Ready, 1, 11) == 2000 && book.Take(StampBook::Done, 0, 10) == 0); // done 10 is not there yet
        book.Put(StampBook::Done, 1, 11, 2500);
        CHECK(book.Take(StampBook::Done, 1, 11) == 2500 && book.Take(StampBook::Done, 0, 10) == 0);  // done 11 before done 10
        book.Put(StampBook::Done, 0, 10, 3000);
        CHECK(book.Take(StampBook::Done, 0, 10) == 3000 && book.Take(StampBook::Ready, 0, 10) == 1000);
        // the slot is used again (4 slots) by frame 14: the old values are gone
        book.Put(StampBook::Ready, 4, 14, 9000);
        CHECK(book.Take(StampBook::Ready, 0, 10) == 0 && book.Take(StampBook::Ready, 0, 14) == 9000 &&
              book.Take(StampBook::Done, 0, 14) == 0);
        CHECK(book.Take(StampBook::Ready, 2, 0) == 0); // no value, no stamp
    }

    // a stamp's key carries the path the value came from: the same value on the other path is another frame
    {
        CHECK(StampKey(7, false) == 7 && StampKey(7, true) != StampKey(7, false));
        StampBook book;
        book.Put(StampBook::Ready, 0, StampKey(7, false), 100);
        CHECK(book.Take(StampBook::Ready, 0, StampKey(7, false)) == 100 && book.Take(StampBook::Ready, 0, StampKey(7, true)) == 0);
    }

    // the done stamp: only a generated frame of a batch with a lead waits for it
    CHECK(DoneStampWaitMs(0, 1, 2) == 0 && DoneStampWaitMs(30000, 1, 2) > 0 && DoneStampWaitMs(30000, 2, 2) > 0);
    CHECK(DoneStampWaitMs(30000, 3, 2) == 0 && DoneStampWaitMs(30000, 0, 2) == 0); // the game's own frame, a bad k

    // V-Sync off: MAILBOX becomes IMMEDIATE when frame generation is on, the surface offers it, the setting and the game's list allow
    {
        const auto tear = ChoosePresentMode(1, true, true, true, true);
        CHECK(tear.mode == 0 && tear.why == TearingWhy::Tearing);
        CHECK(ChoosePresentMode(1, true, true, true, false).mode == 1 && ChoosePresentMode(1, true, true, true, false).why == TearingWhy::FgOff);
        CHECK(ChoosePresentMode(1, true, true, false, true).mode == 1 && ChoosePresentMode(1, true, true, false, true).why == TearingWhy::SettingOff);
        CHECK(ChoosePresentMode(1, false, true, true, true).mode == 1 && ChoosePresentMode(1, false, true, true, true).why == TearingWhy::NoImmediate);
        CHECK(ChoosePresentMode(1, true, false, true, true).mode == 1 &&
              ChoosePresentMode(1, true, false, true, true).why == TearingWhy::ListLacksImmediate);

        for (int mode : { 0, 2, 3 }) // IMMEDIATE, FIFO, FIFO_RELAXED stay whatever else is true
            CHECK(ChoosePresentMode(mode, true, true, true, true).mode == mode && ChoosePresentMode(mode, true, true, true, true).why == TearingWhy::NotMailbox);

        CHECK(ChainedPresentMode(true, 1) == 0 && ChainedPresentMode(true, 2) == 2 && ChainedPresentMode(false, 1) == 1);
        // the swapchain's real mode is IMMEDIATE: timed on the CPU, not display paced, the gate bound for a timed batch (3)
        CHECK(!VsyncPaces(tear.mode) && !DisplaySpaces(true, tear.mode, 0) && LatencyBound(3, 4, VsyncPaces(tear.mode)) == 3);
        CHECK(std::string(TearingText(TearingWhy::ListLacksImmediate)).find("scaling") != std::string::npos);
        CHECK(TearingText(TearingWhy::Tearing) != nullptr && TearingText(TearingWhy::NoImmediate) != nullptr);
    }

    // the pacer's own interval from the batch it planned (step = interval / (generated + 1))
    CHECK(std::fabs(PacerIntervalMs(55556, 2, 10000000) - 16.6668) < 0.01 && PacerIntervalMs(0, 2, 10000000) == 0.0);

    // the game thread's budget
    CHECK(RemainingMs(50.0, 0.0) == 50);
    CHECK(RemainingMs(50.0, 20.5) == 29);
    CHECK(RemainingMs(50.0, 49.5) == 1);
    CHECK(RemainingMs(50.0, 80.0) == 1);

    if (fails == 0)
        std::printf("f5low_vk_rules_smoke: PASS\n");
    return fails == 0 ? 0 : 1;
}
