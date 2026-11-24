// x64 MSVC: cl /std:c++20 /EHsc /W4 /WX /O2 tests/status_page_rules_smoke.cpp
// The words the menu's Status page shows for each state (OptiScaler/menu/StatusPageRules.h).
#include "../OptiScaler/menu/StatusPageRules.h"

#include <cstdio>
#include <string>

using namespace StatusPageRules;

static int failures = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

static bool Is(const std::string& value, const char* expected) { return value == expected; }

int main()
{
    // API, with the translation layer
    CHECK(Is(GameApiText(Api::DX12, false, false), "DX12"));
    CHECK(Is(GameApiText(Api::DX11, true, false), "DX11 (DXVK)"));
    CHECK(Is(GameApiText(Api::DX12, false, true), "DX12 (VKD3D)"));
    CHECK(Is(GameApiText(Api::DX12, true, true), "DX12 (DXVK)")); // DXVK named first, as upstream
    CHECK(Is(GameApiText(Api::NotSelected, true, false), "-"));   // nothing to qualify before the API is known
    CHECK(Is(ApiName(Api::Vulkan), "Vulkan"));

    // Frame generation names, every value
    CHECK(Is(FgInputName(FgInput::NoFG), "Off"));
    CHECK(Is(FgInputName(FgInput::Upscaler), "Upscaler"));
    CHECK(Is(FgInputName(FgInput::DLSSG), "DLSSG"));
    CHECK(Is(FgInputName(FgInput::NvngxFG), "Nvngx FG"));
    CHECK(Is(FgInputName(FgInput::FSRFG), "FSR FG"));
    CHECK(Is(FgInputName(FgInput::FSRFG30), "FSR FG 3.0"));
    CHECK(Is(FgInputName(FgInput::XeFG), "XeFG"));
    CHECK(Is(FgInputName(FgInput::ForceXeLL), "XeLL only"));
    CHECK(Is(FgOutputName(FgOutput::NoFG), "Off"));
    CHECK(Is(FgOutputName(FgOutput::F5Low), "F5Low"));
    CHECK(Is(FgOutputName(FgOutput::DLSSG), "DLSSG"));
    CHECK(Is(FgOutputName(FgOutput::FSRFG), "FSR FG"));
    CHECK(Is(FgOutputName(FgOutput::XeFG), "XeFG"));

    // Nvngx replacement: Custom shows its folder, an empty folder name falls back
    CHECK(Is(FgNvngxName(FgNvngx::None, ""), "Off"));
    CHECK(Is(FgNvngxName(FgNvngx::Nukems, ""), "Nukems"));
    CHECK(Is(FgNvngxName(FgNvngx::Arturs, ""), "Enabler"));
    CHECK(Is(FgNvngxName(FgNvngx::FFX, ""), "FFX"));
    CHECK(Is(FgNvngxName(FgNvngx::Custom, "Apex Neural Rendering FG"), "Apex Neural Rendering FG"));
    CHECK(Is(FgNvngxName(FgNvngx::Custom, ""), "Custom"));
    CHECK(Is(FgNvngxName(FgNvngx::Combo, ""), "Combo"));

    // Ratio override: all presets wins over per preset
    CHECK(Is(RatioOverrideText(false, 1.5f, false), "Off"));
    CHECK(Is(RatioOverrideText(false, 1.5f, true), "Per preset"));
    CHECK(Is(RatioOverrideText(true, 1.5f, true), "All 1.50"));

    // Memory protection: the ini value, and whether it runs
    CHECK(Is(ProtectMemoryText(0, false), "Off"));
    CHECK(Is(ProtectMemoryText(1, true), "F5Low"));
    CHECK(Is(ProtectMemoryText(2, true), "All of OptiScaler's"));
    CHECK(Is(ProtectMemoryText(1, false), "F5Low (not active)"));
    CHECK(Is(ProtectMemoryText(7, true), "?"));

    // HDR output: the Image page's line without its prefix, no line is off
    CHECK(Is(HdrOutputText(false, "anything"), "Off"));
    CHECK(Is(HdrOutputText(true, "HDR output: active, HDR10, 1000 nits"), "active, HDR10, 1000 nits"));
    CHECK(Is(HdrOutputText(true, "applies at the next start"), "applies at the next start"));

    // Neural Rendering: off wins over a stale failure reason, the backend only once the model loaded
    CHECK(Is(NeuralRenderingText(false, "no model", "NVIDIA NGX"), "Off"));
    CHECK(Is(NeuralRenderingText(true, "", ""), "On, not started yet"));
    CHECK(Is(NeuralRenderingText(true, "", "NVIDIA NGX"), "On (NVIDIA NGX)"));
    CHECK(Is(NeuralRenderingText(true, "no model", "NVIDIA NGX"), "Failed: no model"));

    // Optical F5Low: standing aside only when a mode is on; the Neural Rendering modes wait for Neural Rendering,
    // motion only does not
    CHECK(Is(OpticalF5LowText(NativeMode::Off, true, false), "Off"));
    CHECK(Is(OpticalF5LowText(NativeMode::NrOnly, false, true), "Neural Rendering only"));
    CHECK(Is(OpticalF5LowText(NativeMode::NrOnly, false, false), "Neural Rendering only, waiting for Neural Rendering"));
    CHECK(Is(OpticalF5LowText(NativeMode::NrAndFrameGeneration, true, true),
             "Neural Rendering + frame generation, stands aside (the game calls an upscaler)"));
    CHECK(Is(OpticalF5LowText(NativeMode::NrAndFrameGeneration, true, false),
             "Neural Rendering + frame generation, stands aside (the game calls an upscaler)"));
    CHECK(Is(OpticalF5LowText(NativeMode::MotionOnly, false, false), "Motion only"));

    // Neural Rendering settings
    CHECK(Is(NrModelText(0, "", ""), "DLL's choice"));
    CHECK(Is(NrModelText(1, "", ""), "Preset A (main)"));
    CHECK(Is(NrModelText(2, "", "StrongB-NBA1"), "Preset B (small), weights A: stock, B: StrongB-NBA1"));
    CHECK(Is(NrModelText(9, "", ""), "DLL's choice"));
    CHECK(Is(NrPassesText(1, 1.0f), "1 pass, working scale 1.00"));
    CHECK(Is(NrPassesText(3, 0.85f), "3 passes, working scale 0.85"));
    CHECK(Is(NrPlacementText(false, false, false), "After Super Resolution"));
    CHECK(Is(NrPlacementText(false, true, false), "Before Super Resolution"));
    CHECK(Is(NrPlacementText(true, true, false), "Finished Picture")); // the combo's rule: Finished Picture wins
    CHECK(Is(NrPlacementText(false, true, true), "Generate before Super Resolution, apply after"));
    CHECK(Is(NrDetailReuseText(false, 3), "Off"));
    CHECK(Is(NrDetailReuseText(true, 1), "Off"));
    CHECK(Is(NrDetailReuseText(true, 2), "Model every 2nd frame"));
    CHECK(Is(NrDetailReuseText(true, 3), "Model every 3rd frame"));
    CHECK(Is(NrDetailReuseText(true, 4), "Model every 4th frame"));

    using DlssNrVulkanRoute::RunsOn;
    CHECK(Is(VulkanRouteText(true, false, false, true, "", RunsOn::D3D12Bridge),
             "Bridge, game's upscaler (Neural Rendering on D3D12)"));
    CHECK(Is(VulkanRouteText(false, false, true, false, "dlss_12", RunsOn::D3D12GameCopy),
             "Native swapchain, but the upscaler is still dlss_12 (Neural Rendering on D3D12)"));
    CHECK(Is(VulkanRouteText(false, false, false, false, "", RunsOn::Vulkan),
             "Native swapchain (Neural Rendering on Vulkan)"));

    if (failures == 0)
        printf("status_page_rules_smoke: PASS\n");

    return failures == 0 ? 0 : 1;
}
