// Host check of menu/PresentRateMeter.h: the overlay's Vulkan FG output readout averages the last 100 output presents
// (as the D3D12 overlay averages its last 100 drawn frames), so a steady batch cadence reads steady.
// cl /std:c++20 /EHsc /W4 tests\present_rate_meter_smoke.cpp
#include "../OptiScaler/menu/PresentRateMeter.h"

#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                                                          \
            ++g_fail;                                                                                                  \
        }                                                                                                              \
    } while (0)

// A game at gameMs per frame, n shown per game frame spaced spacingMs apart; FrameTime read once per game frame
static void Run(double gameMs, int n, double spacingMs, double expectMs, double tolMs)
{
    PresentRateMeter m;
    m.SetEnabled(true);
    double minFt = 1e9, maxFt = 0;
    for (int f = 0; f < 600; ++f)
    {
        const double t0 = f * gameMs;
        for (int k = 0; k < n; ++k)
            m.Present(t0 + 1.0 + k * spacingMs);
        const double ft = m.FrameTime(t0 + 0.5 + gameMs * 0.37); // menu draws at an odd point of the frame
        if (f > 100 && ft > 0)
        {
            minFt = std::min(minFt, ft);
            maxFt = std::max(maxFt, ft);
        }
    }
    std::printf("game %.1f ms x%d: readout %.3f..%.3f ms (expect %.3f)\n", gameMs, n, minFt, maxFt, expectMs);
    CHECK(std::fabs(minFt - expectMs) <= tolMs && std::fabs(maxFt - expectMs) <= tolMs);
}

int main()
{
    Run(40.0, 5, 8.0, 8.0, 0.35);       // RDR2: 25 x 5 = 125 (old 125 ms window read 117..128 = 7.8..8.5 ms)
    Run(50.0, 6, 8.333, 8.333, 0.45);   // 20 x 6 = 120
    Run(16.667, 2, 8.333, 8.333, 0.15); // 60 x 2
    Run(100.0, 2, 33.3, 50.0, 3.0);     // low rate: bounded by MaxWindowMs, still the whole-frame average
    std::printf(g_fail ? "present_rate_meter_smoke: FAIL\n" : "present_rate_meter_smoke: PASS\n");
    return g_fail ? 1 : 0;
}
