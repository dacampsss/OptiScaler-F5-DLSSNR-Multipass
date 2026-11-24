// Host check of DlssNr_VulkanRoute.h: what the menu says about a Vulkan game's route. No GPU and no game needed.
// cl /std:c++20 /EHsc /W4 /WX tests/vulkan_route_rule_smoke.cpp
#include "../OptiScaler/dlssnr/DlssNr_VulkanRoute.h"

#include <cstdio>

using namespace DlssNrVulkanRoute;

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

int main()
{
    // Which outputs need the bridge: the two D3D12-only libraries
    CHECK(!OutputNeedsBridge(Output::NoFG));
    CHECK(!OutputNeedsBridge(Output::FSRFG));
    CHECK(!OutputNeedsBridge(Output::DLSSG));
    CHECK(OutputNeedsBridge(Output::XeFG));
    CHECK(!OutputNeedsBridge(Output::F5Low));

    // The combo's label carries the note only on the native route, only on those two
    CHECK(OutputLabel(Output::XeFG, true) == "XeFG (bridge)");
    CHECK(OutputLabel(Output::F5Low, true) == "F5Low");
    CHECK(OutputLabel(Output::XeFG, false) == "XeFG");
    CHECK(OutputLabel(Output::F5Low, false) == "F5Low");
    CHECK(OutputLabel(Output::FSRFG, true) == "FSR FG");
    CHECK(OutputLabel(Output::DLSSG, true) == "DLSSG");
    CHECK(OutputLabel(Output::NoFG, true) == "None");

    // The native route: offered, route off, Optical F5Low's frame generation off
    CHECK(NativeRoute(true, false, false));
    CHECK(!NativeRoute(true, true, false));
    CHECK(!NativeRoute(true, false, true));
    CHECK(!NativeRoute(false, false, false)); // a D3D game or dxvk: no route at all

    // The offer: only for XeFG on the native route (F5Low presents there itself); the route needs an upscaler call, else F5Low
    for (bool calls : { false, true })
    {
        CHECK(BridgeOffer(Output::FSRFG, true, calls) == Offer::None);
        CHECK(BridgeOffer(Output::DLSSG, true, calls) == Offer::None);
        CHECK(BridgeOffer(Output::NoFG, true, calls) == Offer::None);
        CHECK(BridgeOffer(Output::XeFG, false, calls) == Offer::None);
        CHECK(BridgeOffer(Output::F5Low, false, calls) == Offer::None);
    }
    CHECK(BridgeOffer(Output::XeFG, true, true) == Offer::UseBridge);
    CHECK(BridgeOffer(Output::F5Low, true, true) == Offer::None);
    CHECK(BridgeOffer(Output::XeFG, true, false) == Offer::UseF5Low);
    CHECK(BridgeOffer(Output::F5Low, true, false) == Offer::None);
    CHECK(BridgeNeededLine(Output::XeFG) == "XeFG runs on D3D12: it needs the Vulkan bridge.");

    // Where Neural Rendering runs
    auto runs = [](bool nr, bool route, bool bridge, bool on12, bool motion, bool fp, bool fpRuns, bool rVk, bool rD3d)
    {
        NrInputs in;
        in.nrEnabled = nr;
        in.routeOn = route;
        in.bridgeUp = bridge;
        in.upscalerOnD3D12 = on12;
        in.motionModeOn = motion;
        in.finishedPicture = fp;
        in.finishedPictureRunsOnVulkan = fpRuns;
        in.runningVk = rVk;
        in.runningD3D12 = rD3d;
        return NrRunsOn(in);
    };
    //                nr    route bridge on12  motion fp     fpRuns rVk    rD3d
    CHECK(runs(false, true, true, true, true, true, true, true, true) == RunsOn::NrOff);
    CHECK(runs(true, false, false, false, false, false, false, false, false) == RunsOn::Vulkan);
    CHECK(runs(true, true, true, false, false, false, false, false, false) == RunsOn::D3D12Bridge);
    CHECK(runs(true, true, false, false, false, false, false, false, false) == RunsOn::D3D12GameCopy); // not up yet
    CHECK(runs(true, false, false, true, false, false, false, false, false) == RunsOn::D3D12GameCopy);  // dlss_12 left
    CHECK(runs(true, false, true, false, true, true, false, false, false) == RunsOn::D3D12F5Low);       // NR only / + FG
    CHECK(runs(true, false, false, true, true, false, false, false, false) == RunsOn::D3D12F5Low);      // beats on12
    // Finished Picture on native Vulkan: on the private D3D12 device while it runs, "not running" while it waits or is
    // refused (and without Vulkan support in a build)
    CHECK(runs(true, false, false, false, false, true, false, false, false) == RunsOn::FinishedPictureNotRunning);
    CHECK(runs(true, false, false, false, false, true, true, false, false) == RunsOn::FinishedPictureNotRunning);
    // The runtime wins over the settings
    CHECK(runs(true, true, true, false, false, false, false, true, false) == RunsOn::Vulkan);
    CHECK(runs(true, false, false, false, false, true, false, false, true) == RunsOn::D3D12);
    CHECK(runs(true, false, false, false, false, false, false, false, true) == RunsOn::D3D12);
    CHECK(runs(true, false, false, false, false, true, true, false, true) == RunsOn::D3D12FinishedPicture);

    // Finished Picture with an F5Low output (the Optical F5Low motion setting on beside it): when Finished Picture ran, it is the
    // one running Neural Rendering; before that, or when it was refused, the setting says what was meant
    {
        NrInputs in;
        in.nrEnabled = true;
        in.motionModeOn = true;
        in.finishedPicture = true;
        in.runningD3D12 = true;
        in.finishedPictureRan = true;
        CHECK(NrRunsOn(in) == RunsOn::D3D12FinishedPicture);
        in.finishedPictureRan = false;
        CHECK(NrRunsOn(in) == RunsOn::D3D12F5Low);
        in.finishedPictureRan = true;
        in.finishedPictureRunsOnVulkan = false;
        CHECK(NrRunsOn(in) == RunsOn::D3D12F5Low);
        in.finishedPictureRunsOnVulkan = true;
        in.finishedPicture = false;
        CHECK(NrRunsOn(in) == RunsOn::D3D12F5Low);
        in.finishedPicture = true;
        in.routeOn = true; // the route turns Finished Picture off: the route's line stays
        in.bridgeUp = true;
        CHECK(NrRunsOn(in) == RunsOn::D3D12Bridge);
    }

    CHECK(RunsOnText(RunsOn::Vulkan) == "Vulkan (native swapchain)");
    CHECK(RunsOnText(RunsOn::D3D12Bridge) == "D3D12 (Vulkan bridge)");
    CHECK(RunsOnText(RunsOn::D3D12GameCopy) == "D3D12 (game's upscaler copy)");
    CHECK(RunsOnText(RunsOn::D3D12F5Low) == "D3D12 (Optical F5Low, copies)");
    CHECK(RunsOnText(RunsOn::D3D12FinishedPicture) == "D3D12 (Finished Picture, copies)");
    CHECK(RunsOnText(RunsOn::D3D12) == "D3D12");
    CHECK(CardRunsOnLine(RunsOn::NrOff) == "Now: Neural Rendering is off.");
    CHECK(CardRunsOnLine(RunsOn::Vulkan) == "Now: Neural Rendering runs on Vulkan, in the game's own device.");
    CHECK(CardRunsOnLine(RunsOn::D3D12Bridge) == "Now: Neural Rendering runs on D3D12, through the bridge.");
    CHECK(CardRunsOnLine(RunsOn::D3D12F5Low) ==
          "Now: Neural Rendering runs on D3D12, because Optical F5Low copies the picture there.");
    CHECK(CardRunsOnLine(RunsOn::D3D12FinishedPicture) ==
          "Now: Neural Rendering runs on D3D12, because Finished Picture copies the picture there.");
    CHECK(CardRunsOnLine(RunsOn::D3D12GameCopy) ==
          "Now: Neural Rendering runs on D3D12, because the game's upscaler still runs on its D3D12 copy.");
    CHECK(CardRunsOnLine(RunsOn::FinishedPictureNotRunning) ==
          "Now: Neural Rendering is not running (Finished Picture is waiting, see its status).");
    CHECK(CardRunsOnLine(RunsOn::D3D12) == "Now: Neural Rendering runs on D3D12.");
    CHECK(PageRunsOnLine(RunsOn::D3D12F5Low) == "Runs on: D3D12 (Optical F5Low, copies)");

    // The leftover: only dlss_12 (the route's own), route off, no F5Low mode, a Vulkan game only
    CHECK(LeftoverD3D12Upscaler(true, false, false, true));
    CHECK(!LeftoverD3D12Upscaler(false, false, false, true));
    CHECK(!LeftoverD3D12Upscaler(true, true, false, true));
    CHECK(!LeftoverD3D12Upscaler(true, false, true, true));
    CHECK(!LeftoverD3D12Upscaler(true, false, false, false)); // ffx_12 / fsr21_12 are the player's choice
    CHECK(LeftoverLine("dlss_12") ==
          "The game's upscaler runs on its D3D12 copy (dlss_12), so Neural Rendering runs on D3D12.");

    CHECK(std::string(NativeSummary()) == "Native swapchain: frame generation with FSR FG, DLSS-G or F5Low. Lighter than the bridge.");
    CHECK(std::string(NativeSummary()).find("FSR FG, DLSS-G or F5Low") != std::string::npos);
    CHECK(std::string(BridgeSummary()) == "Bridge: adds XeFG. Costs a second (D3D12) device and picture copies.");
    CHECK(std::string(BridgeSummary()).find("adds XeFG") != std::string::npos);

    // The Status page's row
    CHECK(StatusText(true, false, false, true, "", RunsOn::D3D12Bridge) ==
          "Bridge, game's upscaler (Neural Rendering on D3D12)");
    CHECK(StatusText(false, true, false, true, "", RunsOn::D3D12F5Low) ==
          "Bridge, Optical F5Low NR + frame generation (Neural Rendering on D3D12, Optical F5Low copies)");
    CHECK(StatusText(false, false, true, false, "dlss_12", RunsOn::D3D12GameCopy) ==
          "Native swapchain, but the upscaler is still dlss_12 (Neural Rendering on D3D12)");
    CHECK(StatusText(false, false, false, true, "", RunsOn::Vulkan) ==
          "Native swapchain after a restart (the bridge is still up) (Neural Rendering on Vulkan)");
    CHECK(StatusText(false, false, false, false, "", RunsOn::Vulkan) == "Native swapchain (Neural Rendering on Vulkan)");
    CHECK(StatusText(false, false, false, false, "", RunsOn::NrOff) == "Native swapchain (Neural Rendering off)");
    CHECK(StatusText(false, false, false, false, "", RunsOn::D3D12FinishedPicture) ==
          "Native swapchain (Neural Rendering on D3D12, Finished Picture copies)");
    CHECK(StatusText(false, false, false, false, "", RunsOn::FinishedPictureNotRunning) ==
          "Native swapchain (Finished Picture is not running)");

    if (fails == 0)
        printf("vulkan_route_rule_smoke: PASS\n");

    return fails == 0 ? 0 : 1;
}
