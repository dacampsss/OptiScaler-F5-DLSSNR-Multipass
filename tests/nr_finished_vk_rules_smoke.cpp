// Host check of dlssnr/DlssNr_FinishedVkRules.h: the capture ring of Finished Picture on a native Vulkan game and the
// formats its capture pass reads. No GPU and no game.
//
//   cl /std:c++20 /EHsc /W4 /Iexternal\vulkan\include tests\nr_finished_vk_rules_smoke.cpp
#include "../OptiScaler/dlssnr/DlssNr_FinishedVkRules.h"
#include <string>

#include <cstdio>

using namespace DlssNr::FinishedVkRules;
using DlssNr::FinishedGameFrame;

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

static FinishedGameFrame Frame(uint64_t epoch, bool reset = false, uint32_t w = 1920, uint32_t h = 1080)
{
    FinishedGameFrame f;
    f.captureEpoch = epoch;
    f.reset = reset;
    f.outputWidth = w;
    f.outputHeight = h;
    return f;
}

static uint64_t g_clock = 0;

// Begin that must find a slot
static size_t Beg(GuideRing& ring, uint64_t cmd, uint64_t now, const FinishedGameFrame& frame)
{
    size_t index = 0;
    CHECK(ring.Begin(cmd, now, frame, index));
    return index;
}

// A successful submit of `cmd` on queue 1
static bool Sub(GuideRing& ring, uint64_t cmd) { return ring.Submitted(cmd, 1, ++g_clock, 0); }

// A present with every submit so far visible and every capture ready
static bool P(GuideRing& ring, uint64_t epoch, uint32_t w, uint32_t h, size_t& index, bool& reset)
{
    return ring.Pick(epoch, UINT64_MAX, w, h, index, reset, [](const GuideRing::Slot&) { return true; }) ==
           PickResult::Ready;
}

int main()
{
    // ---- formats ----
    CHECK(ReadMotion(VK_FORMAT_R16G16_SFLOAT) == GuideRead::View);
    CHECK(ReadMotion(VK_FORMAT_R32G32_SFLOAT) == GuideRead::View);
    CHECK(ReadMotion(VK_FORMAT_R16G16B16A16_SFLOAT) == GuideRead::View);
    CHECK(ReadMotion(VK_FORMAT_R16G16_SNORM) == GuideRead::View);
    CHECK(ReadMotion(VK_FORMAT_R16G16_SINT) == GuideRead::Refused);   // integer: no float sample
    CHECK(ReadMotion(VK_FORMAT_R32_SFLOAT) == GuideRead::Refused);    // one channel is no vector
    CHECK(ReadMotion(VK_FORMAT_D32_SFLOAT) == GuideRead::Refused);
    CHECK(ReadMotion(VK_FORMAT_B10G11R11_UFLOAT_PACK32) == GuideRead::Refused);

    CHECK(ReadDepth(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT) == GuideRead::View);
    CHECK(ReadDepth(VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT) == GuideRead::View);
    CHECK(ReadDepth(VK_FORMAT_X8_D24_UNORM_PACK32, VK_IMAGE_ASPECT_DEPTH_BIT) == GuideRead::View);
    // A view over depth and stencil cannot be sampled: the pass makes a depth-only view
    CHECK(ReadDepth(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ==
          GuideRead::DepthOnly);
    CHECK(ReadDepth(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT) == GuideRead::View);
    CHECK(ReadDepth(VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT) == GuideRead::View); // colour-format depth
    CHECK(ReadDepth(VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT) == GuideRead::Refused);
    CHECK(ReadDepth(VK_FORMAT_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT) == GuideRead::Refused);

    // ---- a recording is not a capture until its command buffer is submitted ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        const size_t slot = Beg(ring, 0x10, 100, Frame(100));
        CHECK(ring.Armed() == 1);
        CHECK(!P(ring, 101, 1920, 1080, index, reset)); // recorded, never submitted: nothing to use
        CHECK(Sub(ring, 0x10));
        CHECK(ring.Armed() == 0);
        CHECK(!Sub(ring, 0x10)); // a second submit of the same buffer is not a second capture
        CHECK(P(ring, 101, 1920, 1080, index, reset));
        CHECK(index == slot);
        CHECK(!P(ring, 101, 1920, 1080, index, reset)); // used once: the same capture does not serve two presents
    }

    // ---- the window: one present after the capture, like the D3D12 slots ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 100, Frame(100));
        Sub(ring, 1);
        CHECK(P(ring, 100, 1920, 1080, index, reset)); // a present at the capture's own count
        Beg(ring, 2, 101, Frame(101));
        Sub(ring, 2);
        CHECK(P(ring, 102, 1920, 1080, index, reset)); // one later
        Beg(ring, 3, 110, Frame(110));
        Sub(ring, 3);
        CHECK(!P(ring, 112, 1920, 1080, index, reset)); // two later: stale, dropped
        CHECK(!P(ring, 111, 1920, 1080, index, reset)); // and gone
        Beg(ring, 4, 120, Frame(120));
        Sub(ring, 4);
        CHECK(!P(ring, 119, 1920, 1080, index, reset)); // a present before the capture's count (race): not this one
    }

    // ---- the newest capture wins, older ones go with it ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        const size_t a = Beg(ring, 1, 200, Frame(200));
        const size_t b = Beg(ring, 2, 200, Frame(200)); // the game records a frame ahead
        Sub(ring, 1);
        Sub(ring, 2);
        CHECK(a != b);
        CHECK(P(ring, 201, 1920, 1080, index, reset));
        CHECK(index == b);
        CHECK(!P(ring, 201, 1920, 1080, index, reset)); // the older one was used up with it
    }

    // ---- size: a capture for another output size is not this picture's ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 300, Frame(300, false, 960, 540));
        Sub(ring, 1);
        CHECK(!P(ring, 301, 1920, 1080, index, reset));
        CHECK(P(ring, 301, 960, 540, index, reset)); // still there for the swapchain it fits
    }

    // ---- a command buffer recorded over: the earlier recording of it is gone (the begin hook says so) ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 7, 400, Frame(400));
        CHECK(ring.Forget(7)); // the game began buffer 7 again before it was ever submitted
        Beg(ring, 7, 401, Frame(401));
        CHECK(ring.Armed() == 1);
        Sub(ring, 7);
        CHECK(P(ring, 402, 1920, 1080, index, reset));
        CHECK(ring.At(index).frame.captureEpoch == 401);
    }

    // ---- two upscaler calls in one command buffer (two sizes): both go with the one submit ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 8, 410, Frame(410, false, 1920, 1080));
        Beg(ring, 8, 410, Frame(410, false, 640, 360));
        CHECK(ring.Armed() == 2);
        Sub(ring, 8);
        CHECK(ring.Armed() == 0);
        CHECK(P(ring, 411, 1920, 1080, index, reset)); // each picture finds the capture of its own size
        CHECK(P(ring, 411, 640, 360, index, reset));
        CHECK(ring.At(index).frame.outputWidth == 640);
    }

    // ---- reset / free of an unsent buffer; a submitted one is not touched ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 500, Frame(500));
        Beg(ring, 2, 500, Frame(500));
        Sub(ring, 2);
        CHECK(ring.Forget(1));
        CHECK(ring.Armed() == 0);
        CHECK(!ring.Forget(2)); // already submitted: what it recorded runs
        CHECK(P(ring, 501, 1920, 1080, index, reset));
    }

    // ---- a secondary command buffer: the submit carries its primary ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 0x20, 600, Frame(600)); // recorded into a secondary buffer
        ring.Rearm(0x20, 0x30);            // executed from primary 0x30
        CHECK(!Sub(ring, 0x20));
        CHECK(Sub(ring, 0x30));
        CHECK(P(ring, 601, 1920, 1080, index, reset));
    }

    // ---- unsent for too long: dropped ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 700, Frame(700));
        ring.Expire(700 + kArmedWindow);
        CHECK(ring.Armed() == 1);
        ring.Expire(700 + kArmedWindow + 1);
        CHECK(ring.Armed() == 0);
        CHECK(!Sub(ring, 1));
        CHECK(!P(ring, 701, 1920, 1080, index, reset));
    }

    // ---- slot reuse: a slot is written again only once its capture is kReuseAge presents old ----
    {
        GuideRing ring;
        size_t index = 0;

        for (uint64_t i = 0; i < kSlots; ++i)
            Beg(ring, 10 + i, 800, Frame(800));

        CHECK(!ring.Begin(99, 800, Frame(800), index));                 // all busy at the same present: skipped
        CHECK(!ring.Begin(99, 800 + kReuseAge - 1, Frame(800), index)); // still too young
        Sub(ring, 10);
        CHECK(!ring.Begin(99, 802, Frame(802), index));                 // armed slots are never reused (unsent recordings)
        for (uint64_t i = 1; i < kSlots; ++i)
            Sub(ring, 10 + i);
        CHECK(ring.Begin(99, 800 + kReuseAge, Frame(804), index));      // old enough and sent: the oldest is taken
        CHECK(index == 0);
        CHECK(!Sub(ring, 10)); // the first recording was already sent; nothing waits for it any more
    }

    // ---- an unsent recording of a slot is never reused, even when old ----
    {
        GuideRing ring;
        size_t index = 0;

        for (uint64_t i = 0; i < kSlots; ++i)
            Beg(ring, 10 + i, 800, Frame(800));

        CHECK(!ring.Begin(99, 800 + kReuseAge + 1, Frame(805), index)); // all armed
        ring.Expire(800 + kArmedWindow + 1);
        CHECK(ring.Begin(99, 800 + kArmedWindow + 1, Frame(807), index)); // dropped as never sent, free again
    }

    // ---- the present's own queue orders its captures; another queue's must have finished ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 900, Frame(900));
        CHECK(ring.Submitted(1, /*queue*/ 5, ++g_clock, /*value*/ 7));
        uint64_t done = 6;
        const auto readyFor = [&](uint64_t presentQueue)
        { return [&done, presentQueue](const GuideRing::Slot& slot) { return slot.queue == presentQueue || done >= slot.value; }; };
        CHECK(ring.Pick(901, UINT64_MAX, 1920, 1080, index, reset, readyFor(9)) == PickResult::NotReady);
        CHECK(ring.Pick(901, UINT64_MAX, 1920, 1080, index, reset, readyFor(5)) == PickResult::Ready); // same queue
        Beg(ring, 2, 905, Frame(905));
        CHECK(ring.Submitted(2, 5, ++g_clock, 8));
        done = 8;
        CHECK(ring.Pick(906, UINT64_MAX, 1920, 1080, index, reset, readyFor(9)) == PickResult::Ready);
        CHECK(ring.At(index).frame.captureEpoch == 905);
    }

    // ---- a not-ready capture is kept for the next present, until it is stale ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 910, Frame(910));
        ring.Submitted(1, 5, ++g_clock, 3);
        const auto never = [](const GuideRing::Slot&) { return false; };
        CHECK(ring.Pick(911, UINT64_MAX, 1920, 1080, index, reset, never) == PickResult::NotReady);
        CHECK(ring.Pick(911, UINT64_MAX, 1920, 1080, index, reset, never) == PickResult::NotReady);
        CHECK(ring.Pick(913, UINT64_MAX, 1920, 1080, index, reset, never) == PickResult::None); // too old: dropped
    }

    // ---- a submit made after the present began is for the next frame ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 920, Frame(920));
        ring.Submitted(1, 1, 100, 0); // frame N, submitted before the present
        Beg(ring, 2, 920, Frame(920));
        ring.Submitted(2, 1, 105, 0); // frame N+1, submitted by another thread while present N runs
        const auto ok = [](const GuideRing::Slot&) { return true; };
        CHECK(ring.Pick(921, /*snapshot*/ 101, 1920, 1080, index, reset, ok) == PickResult::Ready);
        CHECK(ring.At(index).submitSerial == 100);                   // N's guides, not N+1's
        CHECK(ring.Armed() == 0);
        CHECK(ring.Pick(921, 110, 1920, 1080, index, reset, ok) == PickResult::Ready); // the next present gets N+1
        CHECK(ring.At(index).submitSerial == 105);
    }

    // ---- a size mismatch is named ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 930, Frame(930, false, 1280, 720));
        ring.Submitted(1, 1, ++g_clock, 0);
        const auto ok = [](const GuideRing::Slot&) { return true; };
        CHECK(ring.Pick(931, UINT64_MAX, 1920, 1080, index, reset, ok) == PickResult::SizeMismatch);
        CHECK(ring.mismatchWidth == 1280 && ring.mismatchHeight == 720);
        CHECK(ring.Pick(931, UINT64_MAX, 1280, 720, index, reset, ok) == PickResult::Ready);
        CHECK(ring.Pick(931, UINT64_MAX, 1920, 1080, index, reset, ok) == PickResult::None);
    }

    // ---- reuse also waits for the GPU (the slot's timeline value) ----
    {
        GuideRing ring;
        size_t index = 0;
        uint64_t reached = 0;
        const auto done = [&](const GuideRing::Slot& slot) { return slot.value == 0 || reached >= slot.value; };

        for (uint64_t i = 0; i < kSlots; ++i)
        {
            Beg(ring, 10 + i, 1000, Frame(1000));
            ring.Submitted(10 + i, 3, ++g_clock, 100 + i);
        }

        CHECK(!ring.Begin(99, 1000 + kReuseAge, Frame(1004), index, done)); // old enough, GPU not there
        reached = 100;
        CHECK(ring.Begin(99, 1000 + kReuseAge, Frame(1004), index, done));  // the oldest slot's value reached
        CHECK(index == 0);
        CHECK(!ring.Begin(98, 1000 + kReuseAge, Frame(1004), index, done)); // the next one's is not
        CHECK(ring.Waiting(99));
        CHECK(!ring.Waiting(10));
    }

    // ---- a timeline signal that could not be queued drops its captures ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 1100, Frame(1100));
        ring.Submitted(1, 4, ++g_clock, 55);
        CHECK(!ring.DropByValue(4, 54)); // another value: untouched
        CHECK(ring.DropByValue(4, 55));
        CHECK(!P(ring, 1101, 1920, 1080, index, reset));
        Beg(ring, 2, 1101, Frame(1101));
        ring.Submitted(2, 4, ++g_clock, 56);
        ring.DropByValue(4, 56);
        // the dropped slot reports no value, so reuse of it does not wait for a signal that never comes
        CHECK(ring.At(1).value == 0 || ring.At(0).value == 0);
    }

    // ---- the idle release of the shared D3D12 pass ----
    CHECK(MayReleaseIdlePass(20000, 5000, 10000, false));
    CHECK(!MayReleaseIdlePass(14000, 5000, 10000, false)); // not idle long enough
    CHECK(!MayReleaseIdlePass(20000, 5000, 10000, true));  // Optical F5Low's NR or the virtual upscaler can use it

    // ---- colour spaces ----
    CHECK(ColourSpaceTaken(VK_COLOR_SPACE_SRGB_NONLINEAR_KHR));
    CHECK(ColourSpaceTaken(VK_COLOR_SPACE_HDR10_ST2084_EXT));
    CHECK(ColourSpaceTaken(VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT));
    CHECK(!ColourSpaceTaken(VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT));
    CHECK(!ColourSpaceTaken(VK_COLOR_SPACE_HDR10_HLG_EXT));
    CHECK(!ColourSpaceTaken(VK_COLOR_SPACE_DOLBYVISION_EXT));
    CHECK(!ColourSpaceTaken(VK_COLOR_SPACE_BT2020_LINEAR_EXT));
    CHECK(!ColourSpaceTaken(VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT));

    // ---- the game's reset request outlives a capture that was dropped unapplied ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 900, Frame(900, true)); // the game reset its upscaler here
        Sub(ring, 1);
        Beg(ring, 2, 900, Frame(900, false));
        Sub(ring, 2);
        CHECK(P(ring, 901, 1920, 1080, index, reset)); // takes the newer; the older one's reset rides on it
        CHECK(reset);

        Beg(ring, 3, 910, Frame(910, true)); // reset capture goes stale unapplied
        Sub(ring, 3);
        CHECK(!P(ring, 913, 1920, 1080, index, reset));
        Beg(ring, 4, 914, Frame(914, false));
        Sub(ring, 4);
        CHECK(P(ring, 915, 1920, 1080, index, reset));
        CHECK(reset); // not lost
        Beg(ring, 5, 916, Frame(916, false));
        Sub(ring, 5);
        CHECK(P(ring, 917, 1920, 1080, index, reset));
        CHECK(!reset); // and used once
    }

    // ---- an armed (unsent) reset capture that expires also keeps the request ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 1000, Frame(1000, true));
        ring.Expire(1000 + kArmedWindow + 1);
        Beg(ring, 2, 1010, Frame(1010, false));
        Sub(ring, 2);
        CHECK(P(ring, 1011, 1920, 1080, index, reset));
        CHECK(reset);
    }

    // ---- Clear: a new swapchain or device starts the history over ----
    {
        GuideRing ring;
        size_t index = 0;
        bool reset = false;
        Beg(ring, 1, 1100, Frame(1100));
        Sub(ring, 1);
        ring.Clear();
        CHECK(!P(ring, 1101, 1920, 1080, index, reset));
        Beg(ring, 2, 1102, Frame(1102));
        Sub(ring, 2);
        CHECK(P(ring, 1103, 1920, 1080, index, reset));
        CHECK(reset);
    }

    // ---- history gap (the D3D12 rule with one present per game frame) ----
    CHECK(HistoryGap(0, 100));         // nothing applied yet
    CHECK(!HistoryGap(100, 101));      // the next frame
    CHECK(!HistoryGap(100, 102));      // one frame missed is still a continuation
    CHECK(HistoryGap(100, 103));       // more than two game frames
    CHECK(!HistoryGap(100, 100));
    CHECK(!HistoryGap(100, 90));       // an earlier capture is no gap

    // ---- frame generation: F5Low keeps the game's swapchain and takes the finished picture, FSR FG / DLSS-G do not ----
    CHECK(BlockedByFrameGeneration(false, false, false) == FgBlock::None);
    CHECK(BlockedByFrameGeneration(true, true, false) == FgBlock::None);   // F5Low output, generating
    CHECK(BlockedByFrameGeneration(false, true, false) == FgBlock::None);  // F5Low output, paused / 1X hold / not active
    CHECK(BlockedByFrameGeneration(true, false, false) == FgBlock::OptiFg); // FSR FG or DLSS-G output
    CHECK(BlockedByFrameGeneration(false, false, false) == FgBlock::None);
    CHECK(BlockedByFrameGeneration(false, false, true) == FgBlock::GameDlssg);  // the game's own DLSS-G
    CHECK(BlockedByFrameGeneration(true, true, true) == FgBlock::GameDlssg);    // even beside F5Low
    CHECK(std::string(FgBlockText(FgBlock::None)).empty());
    CHECK(std::string(FgBlockText(FgBlock::OptiFg)).find("FSR FG or DLSS-G") != std::string::npos);
    CHECK(std::string(FgBlockText(FgBlock::GameDlssg)).find("game's own DLSS") != std::string::npos);

    if (fails == 0)
        printf("nr_finished_vk_rules_smoke: PASS\n");

    return fails == 0 ? 0 : 1;
}
