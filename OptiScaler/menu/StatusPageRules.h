#pragma once

// The words the Status page (menu/StatusPage.cpp) shows for each state. Pure, so a host test runs every value
// (tests/status_page_rules_smoke.cpp); the page itself only reads the state and draws.
//
// The page is ported from upstream OptiScaler's Status tab (TheRazerMD and FakeMichau, GPL-3.0: d63306ea, 434b9555,
// 76511542, 2a382e1f, deb645a1, c220ad9e, 76028322, fc44c2be), drawn with this fork's cards, plus a card for this
// fork's own features.

#include "../dlssnr/DlssNr_VulkanRoute.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace StatusPageRules
{
// Mirrors of the enums in State.h, framegen/FGOutput.h and OptiTypes.h, by value, so the rules need no engine headers.
// StatusPage.cpp checks the values match with static_asserts.
enum class Api : int
{
    NotSelected = 0,
    DX11,
    DX12,
    Vulkan
};

enum class FgInput : uint32_t
{
    NoFG,
    Upscaler,
    DLSSG,
    NvngxFG,
    FSRFG,
    FSRFG30,
    XeFG,
    ForceXeLL
};

enum class FgOutput : uint32_t
{
    NoFG,
    FSRFG,
    DLSSG,
    XeFG,
    F5Low
};

enum class FgNvngx : uint32_t
{
    None,
    Nukems,
    Arturs,
    FFX,
    Combo,
    Custom
};

// dlssnr/DlssNr_NativeMode.h Shown
enum class NativeMode
{
    Off,
    NrOnly,
    NrAndFrameGeneration,
    MotionOnly
};

inline const char* ApiName(Api api)
{
    switch (api)
    {
    case Api::DX11:
        return "DX11";
    case Api::DX12:
        return "DX12";
    case Api::Vulkan:
        return "Vulkan";
    default:
        return "-";
    }
}

// The game's API, with the translation layer it runs through
inline std::string GameApiText(Api api, bool dxvk, bool vkd3dProton)
{
    std::string text = ApiName(api);

    if (api == Api::NotSelected)
        return text;

    if (dxvk)
        text += " (DXVK)";
    else if (vkd3dProton)
        text += " (VKD3D)";

    return text;
}

// The names the Frame Generation page uses
inline const char* FgInputName(FgInput input)
{
    switch (input)
    {
    case FgInput::Upscaler:
        return "Upscaler";
    case FgInput::DLSSG:
        return "DLSSG";
    case FgInput::NvngxFG:
        return "Nvngx FG";
    case FgInput::FSRFG:
        return "FSR FG";
    case FgInput::FSRFG30:
        return "FSR FG 3.0";
    case FgInput::XeFG:
        return "XeFG";
    case FgInput::ForceXeLL:
        return "XeLL only";
    default:
        return "Off";
    }
}

inline const char* FgOutputName(FgOutput output)
{
    switch (output)
    {
    case FgOutput::FSRFG:
        return "FSR FG";
    case FgOutput::DLSSG:
        return "DLSSG";
    case FgOutput::XeFG:
        return "XeFG";
    case FgOutput::F5Low:
        return "F5Low";
    default:
        return "Off";
    }
}

// customName: the custom\<name> folder for Custom
inline std::string FgNvngxName(FgNvngx replacement, const std::string& customName)
{
    switch (replacement)
    {
    case FgNvngx::None:
        return "Off";
    case FgNvngx::Nukems:
        return "Nukems";
    case FgNvngx::Arturs:
        return "Enabler";
    case FgNvngx::FFX:
        return "FFX";
    case FgNvngx::Combo:
        return "Combo";
    case FgNvngx::Custom:
        return customName.empty() ? "Custom" : customName;
    default:
        return "Off";
    }
}

// [Upscalers] ratio overrides: all presets at one ratio wins over the per preset ratios
inline std::string RatioOverrideText(bool allEnabled, float allValue, bool perPresetEnabled)
{
    if (allEnabled)
    {
        char text[32];
        snprintf(text, sizeof(text), "All %.2f", allValue);
        return text;
    }

    return perPresetEnabled ? "Per preset" : "Off";
}

// [Memory] ProtectMemory (0 off, 1 f5low.dll's memory, 2 everything of ours), read once at start; protecting is
// what runs, false when the hooks it needs did not install
inline std::string ProtectMemoryText(int value, bool protecting)
{
    const char* name = "?";

    switch (value)
    {
    case 0:
        return "Off";
    case 1:
        name = "F5Low";
        break;
    case 2:
        name = "All of OptiScaler's";
        break;
    default:
        break;
    }

    return protecting ? std::string(name) : std::string(name) + " (not active)";
}

// HDR output from the Image page's status line ("HDR output: ..."), shown without its prefix; no line means off
inline std::string HdrOutputText(bool hasLine, const std::string& line)
{
    if (!hasLine)
        return "Off";

    constexpr const char* prefix = "HDR output: ";
    const size_t length = std::char_traits<char>::length(prefix);

    return line.compare(0, length, prefix) == 0 ? line.substr(length) : line;
}

// Neural Rendering: off, failed with its reason, on with the model backend, or on before the model has loaded
inline std::string NeuralRenderingText(bool enabled, const std::string& failure, const std::string& backend)
{
    if (!enabled)
        return "Off";

    if (!failure.empty())
        return "Failed: " + failure;

    return backend.empty() ? "On, not started yet" : "On (" + backend + ")";
}

// The "Vulkan route" row of a Vulkan game (the words are DlssNr_VulkanRoute.h's, as the route card reads them)
inline std::string VulkanRouteText(bool routeOn, bool f5lowFrameGeneration, bool leftover, bool bridgeUp,
                                   const std::string& backendCode, DlssNrVulkanRoute::RunsOn runsOn)
{
    return DlssNrVulkanRoute::StatusText(routeOn, f5lowFrameGeneration, leftover, bridgeUp, backendCode, runsOn);
}

// Optical F5Low's mode, and whether it stands aside because the game calls an upscaler of its own
// [DlssNr] Preset (1 A, 2 B, else the DLL's choice) and WeightsSetA / WeightsSetB (folders under nr_weights, empty =
// the DLL's own). The weights part shows when either model has a set.
inline std::string NrModelText(uint32_t preset, const std::string& weightsSetA, const std::string& weightsSetB)
{
    std::string text = preset == 1 ? "Preset A (main)" : preset == 2 ? "Preset B (small)" : "DLL's choice";

    if (!weightsSetA.empty() || !weightsSetB.empty())
        text += ", weights A: " + (weightsSetA.empty() ? std::string("stock") : weightsSetA) +
                ", B: " + (weightsSetB.empty() ? std::string("stock") : weightsSetB);

    return text;
}

// [DlssNr] Passes and WorkingScale
inline std::string NrPassesText(uint32_t passes, float workingScale)
{
    char text[64];
    snprintf(text, sizeof(text), "%u %s, working scale %.2f", passes, passes == 1 ? "pass" : "passes", workingScale);
    return text;
}

// Where the pass runs, by the "NR Pass at" combo's rule: Finished Picture wins over before Super Resolution
inline const char* NrPlacementText(bool finishedPicture, bool beforeSr, bool generateBeforeApplyAfter)
{
    if (finishedPicture)
        return "Finished Picture";

    if (generateBeforeApplyAfter)
        return "Generate before Super Resolution, apply after";

    return beforeSr ? "Before Super Resolution" : "After Super Resolution";
}

// [DlssNr] DetailReuse: the model runs one frame in Interval, its detail moved along in between
inline std::string NrDetailReuseText(bool enabled, uint32_t interval)
{
    if (!enabled || interval < 2)
        return "Off";

    return "Model every " + std::to_string(interval) + (interval == 2 ? "nd" : interval == 3 ? "rd" : "th") + " frame";
}

// nrUsable: Neural Rendering is on and has not failed, which the Neural Rendering modes need
inline std::string OpticalF5LowText(NativeMode mode, bool gameCallsUpscaler, bool nrUsable)
{
    const char* name = "Off";

    switch (mode)
    {
    case NativeMode::NrOnly:
        name = "Neural Rendering only";
        break;
    case NativeMode::NrAndFrameGeneration:
        name = "Neural Rendering + frame generation";
        break;
    case NativeMode::MotionOnly:
        name = "Motion only";
        break;
    default:
        return name;
    }

    std::string text = name;

    if (gameCallsUpscaler)
        text += ", stands aside (the game calls an upscaler)";
    else if (mode != NativeMode::MotionOnly && !nrUsable)
        text += ", waiting for Neural Rendering";

    return text;
}
} // namespace StatusPageRules
