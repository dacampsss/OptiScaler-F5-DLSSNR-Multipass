// Headless shader test: Windows D3D11 WARP runs the experimental input styles ([DlssNr] InputStyle) on the picture
// the model is shown.
//
// Watercolour (1, DlssNrMode_Watercolor, a Kuwahara filter):
//   1. Flat areas come back unchanged.
//   2. A hard edge stays exactly where it is, unblurred.
//   3. Texture is flattened: its spread falls well below the original's (printed).
//   4. Strength 0 returns the picture bit for bit.
// Structure hint (2, DlssNrMode_StructureHint, a CAS-like local contrast boost on luminance):
//   5. Flat areas unchanged, strength 0 bit for bit.
//   6. Texture spread rises (printed).
//   7. Bounded halos: every pixel's luma stays within its 3x3 neighbourhood's original min and max, widened by a
//      quarter of that range (at full strength).
//   8. Colour kept: a coloured texture's channel ratios are unchanged where no channel reached 1.
//
// Every curve the model can be shown (soft knee, Neutwo, hybrid, HLG, PQ, Linear) and a tone-mapped (passthrough) frame.
//
// Build: cl /nologo /std:c++20 /EHsc /W3 tests\nr_input_style_shader_smoke.cpp /link d3d11.lib d3dcompiler.lib
// Run:   nr_input_style_shader_smoke.exe <dlssnr.hlsl>
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
using Microsoft::WRL::ComPtr;

struct Pixel { float r, g, b, a; };

struct Image
{
    UINT w = 0, h = 0;
    std::vector<Pixel> px;
    Pixel& at(UINT x, UINT y) { return px[y * w + x]; }
    const Pixel& at(UINT x, UINT y) const { return px[y * w + x]; }
};

static void check(HRESULT hr, int line = __builtin_LINE())
{
    if (FAILED(hr)) throw std::runtime_error("D3D call failed, line " + std::to_string(line));
}
static int fails = 0;
static void expect(bool ok, const std::string& label) { if (!ok) { std::printf("FAIL: %s\n", label.c_str()); ++fails; } }

struct Gpu
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<ID3D11SamplerState> linearSampler;

    Gpu()
    {
        check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &ctx));
        D3D11_SAMPLER_DESC sampling {};
        sampling.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampling.AddressU = sampling.AddressV = sampling.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampling.MaxLOD = D3D11_FLOAT32_MAX;
        check(device->CreateSamplerState(&sampling, &linearSampler));
    }

    // tiled: the smooth canvas's own build (DLSSNR_SMOOTH_CANVAS_TILED, 16x16 groups), as D3D12 runs it
    ComPtr<ID3D11ComputeShader> Compile(const wchar_t* path, bool tiled = false)
    {
        ComPtr<ID3DBlob> code, errors;
        // WARP crashes running the shader with the percentile exposure meter in (dlssnr.hlsl); not tested here
        const D3D_SHADER_MACRO plain[] = { { "DLSSNR_NO_PERCENTILE_METER", "1" }, { nullptr, nullptr } };
        const D3D_SHADER_MACRO canvas[] = { { "DLSSNR_NO_PERCENTILE_METER", "1" }, { "DLSSNR_SMOOTH_CANVAS_TILED", "1" },
                                            { nullptr, nullptr } };
        const D3D_SHADER_MACRO* defines = tiled ? canvas : plain;
        HRESULT hr = D3DCompileFromFile(path, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, "CSMain", "cs_5_0",
                                        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (errors && FAILED(hr)) std::fprintf(stderr, "%s", (char*) errors->GetBufferPointer());
        check(hr);
        ComPtr<ID3D11ComputeShader> shader;
        check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader));
        return shader;
    }

    ComPtr<ID3D11Texture2D> Texture(UINT w, UINT h, const Image* data)
    {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = w; desc.Height = h; desc.MipLevels = 1; desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        D3D11_SUBRESOURCE_DATA init {data ? data->px.data() : nullptr, (UINT) (w * sizeof(Pixel)), 0};
        ComPtr<ID3D11Texture2D> texture;
        check(device->CreateTexture2D(&desc, data ? &init : nullptr, &texture));
        return texture;
    }

    Image Read(ID3D11Texture2D* texture)
    {
        D3D11_TEXTURE2D_DESC desc {};
        texture->GetDesc(&desc);
        desc.BindFlags = 0; desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        check(device->CreateTexture2D(&desc, nullptr, &staging));
        ctx->CopyResource(staging.Get(), texture);
        D3D11_MAPPED_SUBRESOURCE mapped {};
        check(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        Image image {desc.Width, desc.Height, std::vector<Pixel>(desc.Width * desc.Height)};
        for (UINT y = 0; y < desc.Height; ++y)
            std::memcpy(&image.px[y * desc.Width], (char*) mapped.pData + y * mapped.RowPitch, desc.Width * sizeof(Pixel));
        ctx->Unmap(staging.Get(), 0);
        return image;
    }

    // One compute dispatch over w x h in group x group thread groups (8 for dlssnr.hlsl's CSMain, 16 for the tiled smooth
    // canvas): t0 (and t1..t4 = t0), u0 target, u1 spare.
    template <typename Constants>
    void Run(ID3D11ComputeShader* shader, const Constants& k, ID3D11Texture2D* t0, ID3D11Texture2D* u0, ID3D11Texture2D* u1,
             UINT w, UINT h, UINT group = 8)
    {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = (UINT) ((sizeof(Constants) + 15) / 16 * 16);
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        std::vector<uint8_t> bytes(desc.ByteWidth);
        std::memcpy(bytes.data(), &k, sizeof(Constants));
        D3D11_SUBRESOURCE_DATA init {bytes.data(), 0, 0};
        ComPtr<ID3D11Buffer> constants;
        check(device->CreateBuffer(&desc, &init, &constants));
        ComPtr<ID3D11ShaderResourceView> s0;
        check(device->CreateShaderResourceView(t0, nullptr, &s0));
        ComPtr<ID3D11UnorderedAccessView> v0, v1;
        check(device->CreateUnorderedAccessView(u0, nullptr, &v0));
        if (u1) check(device->CreateUnorderedAccessView(u1, nullptr, &v1));
        ID3D11ShaderResourceView* srvs[] = {s0.Get(), s0.Get(), s0.Get(), s0.Get(), s0.Get()};
        ID3D11UnorderedAccessView* uavs[] = {v0.Get(), v1.Get()};
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetShaderResources(0, 5, srvs);
        ctx->CSSetUnorderedAccessViews(0, u1 ? 2 : 1, uavs, nullptr);
        ctx->CSSetConstantBuffers(0, 1, constants.GetAddressOf());
        ctx->CSSetSamplers(0, 1, linearSampler.GetAddressOf());
        ctx->Dispatch((w + group - 1) / group, (h + group - 1) / group, 1);
        ID3D11ShaderResourceView* noSrvs[5] = {};
        ID3D11UnorderedAccessView* noUavs[2] = {};
        ctx->CSSetShaderResources(0, 5, noSrvs);
        ctx->CSSetUnorderedAccessViews(0, 2, noUavs, nullptr);
    }
};

static DlssNrConstants Base(uint32_t mode, UINT w, UINT h, uint32_t passthrough, uint32_t reversible)
{
    DlssNrConstants k {};
    k.Mode = mode;
    k.Width = w; k.Height = h;
    k.WhitePoint = 1.0f;
    k.TransferStrength = 1.0f; k.ColourStrength = 1.0f;
    k.MaxRatio = 8.0f;
    k.Passthrough = passthrough;
    k.ApplyModel = 1;
    k.ReversibleMode = reversible;
    k.DebugScale = 1.0f;
    k.ModelWorkScale = 1.0f;
    k.PassFeedback = 1.0f;
    return k;
}

// The encode's proxy (the picture the model would be shown) and its watercolour at a strength.
struct Pictures { Image proxy, painted; };

static Pictures Paint(Gpu& gpu, ID3D11ComputeShader* nr, const Image& frame, uint32_t passthrough, uint32_t reversible,
                      float strength, uint32_t mode = DlssNrMode_Watercolor, bool facesOnly = false)
{
    const UINT w = frame.w, h = frame.h;
    auto source = gpu.Texture(w, h, &frame);
    auto proxy = gpu.Texture(w, h, nullptr);
    auto keep = gpu.Texture(w, h, nullptr);
    auto painted = gpu.Texture(w, h, nullptr);
    gpu.Run(nr, Base(DlssNrMode_Encode, w, h, passthrough, reversible), source.Get(), proxy.Get(), keep.Get(), w, h);
    DlssNrConstants k = Base(mode, w, h, passthrough, reversible);
    k.TransferStrength = strength;
    k.SkinProtection = facesOnly ? 1u : 0u; // the style passes read it as "faces only"
    gpu.Run(nr, k, proxy.Get(), painted.Get(), keep.Get(), w, h);
    return {gpu.Read(proxy.Get()), gpu.Read(painted.Get())};
}

static float Luma(const Pixel& p) { return 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b; }

// 64 x 64: a flat left band, a hard vertical edge (dark | bright) at x = 32 in the middle band, and fine texture
// (pixel grain and a 4-pixel weave) on the right.
static const UINT kSize = 64;

static Image Scene()
{
    Image image {kSize, kSize, std::vector<Pixel>(kSize * kSize)};
    uint32_t seed = 12345;
    for (UINT y = 0; y < kSize; ++y)
        for (UINT x = 0; x < kSize; ++x)
        {
            float v = 0.25f;
            if (x >= 16 && x < 44)
                v = x < 32 ? 0.03f : 0.9f;
            else if (x >= 44)
            {
                seed = seed * 1664525u + 1013904223u;
                const float grain = ((seed >> 8) & 0xFFFF) / 32767.5f - 1.0f;
                const float weave = std::sin(1.5707963f * (x + 0.5f)) * std::sin(1.5707963f * (y + 0.5f) + 0.7f);
                v = 0.3f * (1.0f + 0.2f * (y < kSize / 2 ? grain : weave));
            }
            image.at(x, y) = {v, v, v, 1};
        }
    return image;
}

int wmain(int argc, wchar_t** argv) try
{
    if (argc != 2)
        throw std::runtime_error("Pass the dlssnr.hlsl path");
    Gpu gpu;
    auto nr = gpu.Compile(argv[1]);
    const Image frame = Scene();

    struct Case { uint32_t passthrough, reversible; const char* name; };
    const Case cases[] = {{0, 0, "soft knee"}, {0, 1, "Neutwo"}, {0, 3, "hybrid"}, {0, 5, "HLG"}, {0, 6, "PQ"}, {0, 7, "Linear"},
                          {1, 0, "tone-mapped"}};

    std::printf("%-12s %10s %10s %10s %10s %10s\n", "curve", "flat", "edge", "grain", "weave", "off");
    for (const Case& c : cases)
    {
        const Pictures full = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 1.0f);
        const Pictures none = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 0.0f);

        float flat = 0.0f, edge = 0.0f, off = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
            {
                const Pixel& a = full.proxy.at(x, y);
                const Pixel& b = full.painted.at(x, y);
                const float d = std::max({std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
                if (x >= 3 && x < 13)
                    flat = std::max(flat, d);
                if (x >= 19 && x < 41) // both sides of the edge, including the two pixels touching it
                    edge = std::max(edge, d);
                const Pixel& n = none.painted.at(x, y);
                off = std::max({off, std::abs(n.r - a.r), std::abs(n.g - a.g), std::abs(n.b - a.b)});
            }

        // Texture: standard deviation of the stored luma, after / before, per texture.
        auto spread = [](const Image& im, UINT y0, UINT y1)
        {
            double s = 0.0, s2 = 0.0;
            int n = 0;
            for (UINT y = y0; y < y1; ++y)
                for (UINT x = 47; x < kSize - 3; ++x)
                {
                    const double l = Luma(im.at(x, y));
                    s += l;
                    s2 += l * l;
                    ++n;
                }
            const double m = s / n;
            return std::sqrt(std::max(s2 / n - m * m, 0.0));
        };
        const double grain = spread(full.painted, 3, kSize / 2 - 3) / spread(full.proxy, 3, kSize / 2 - 3);
        const double weave = spread(full.painted, kSize / 2 + 3, kSize - 3) / spread(full.proxy, kSize / 2 + 3, kSize - 3);

        std::printf("%-12s %10.2e %10.2e %9.0f%% %9.0f%% %10.2e\n", c.name, flat, edge, grain * 100.0, weave * 100.0, off);
        const std::string what = std::string(" (") + c.name + ")";
        expect(flat < 1e-4f, "flat areas change" + what);
        expect(edge < 1e-4f, "the hard edge moves or blurs" + what);
        expect(grain < 0.8 && weave < 0.8, "texture is not flattened" + what);
        expect(off == 0.0f, "strength 0 changes the picture" + what);
    }

    // Structure hint.
    std::printf("\n%-12s %10s %10s %10s %10s %10s %10s\n", "curve", "flat", "off", "grain", "weave", "halo", "colour");
    for (const Case& c : cases)
    {
        const Pictures full = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 1.0f, DlssNrMode_StructureHint);
        const Pictures none = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 0.0f, DlssNrMode_StructureHint);

        float flat = 0.0f, off = 0.0f, halo = 0.0f, dark = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
            {
                const Pixel& a = full.proxy.at(x, y);
                const Pixel& b = full.painted.at(x, y);
                if (x >= 3 && x < 13)
                    flat = std::max({flat, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
                const Pixel& n = none.painted.at(x, y);
                off = std::max({off, std::abs(n.r - a.r), std::abs(n.g - a.g), std::abs(n.b - a.b)});
                float lo = 1e30f, hi = -1e30f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const UINT xx = (UINT) std::clamp((int) x + dx, 0, (int) kSize - 1);
                        const UINT yy = (UINT) std::clamp((int) y + dy, 0, (int) kSize - 1);
                        const float l = Luma(full.proxy.at(xx, yy));
                        lo = std::min(lo, l);
                        hi = std::max(hi, l);
                    }
                const float l = Luma(b);
                const float margin = 0.25f * (hi - lo);
                halo = std::max({halo, l - (hi + margin), (lo - margin) - l});
                // The dark side of the hard edge (0.03 | 0.9): no lower than three quarters of the darkest neighbour.
                dark = std::max(dark, 0.75f * lo - l);
            }
        auto spread = [](const Image& im, UINT y0, UINT y1)
        {
            double s = 0.0, s2 = 0.0;
            int n = 0;
            for (UINT y = y0; y < y1; ++y)
                for (UINT x = 47; x < kSize - 3; ++x)
                {
                    const double l = Luma(im.at(x, y));
                    s += l;
                    s2 += l * l;
                    ++n;
                }
            const double m = s / n;
            return std::sqrt(std::max(s2 / n - m * m, 0.0));
        };
        const double grain = spread(full.painted, 3, kSize / 2 - 3) / spread(full.proxy, 3, kSize / 2 - 3);
        const double weave = spread(full.painted, kSize / 2 + 3, kSize - 3) / spread(full.proxy, kSize / 2 + 3, kSize - 3);

        // Colour: a coloured weave; channel ratios before and after, where the result's peak stayed below 1.
        Image coloured = frame;
        for (Pixel& p : coloured.px)
            p = {p.r * 1.0f, p.g * 0.6f, p.b * 0.3f, 1};
        const Pictures tinted = Paint(gpu, nr.Get(), coloured, c.passthrough, c.reversible, 1.0f, DlssNrMode_StructureHint);
        float colour = 0.0f;
        for (size_t i = 0; i < tinted.proxy.px.size(); ++i)
        {
            const Pixel& a = tinted.proxy.px[i];
            const Pixel& b = tinted.painted.px[i];
            if (a.r < 1e-3f || b.r >= 0.9999f)
                continue;
            colour = std::max({colour, std::abs(a.g / a.r - b.g / b.r), std::abs(a.b / a.r - b.b / b.r)});
        }

        std::printf("%-12s %10.2e %10.2e %9.0f%% %9.0f%% %10.2e %10.2e dark %.2e\n", c.name, flat, off, grain * 100.0,
                    weave * 100.0, halo, colour, dark);
        const std::string what = std::string(" (structure hint, ") + c.name + ")";
        expect(flat < 1e-6f, "flat areas change" + what);
        expect(off == 0.0f, "strength 0 changes the picture" + what);
        expect(grain > 1.05 && weave > 1.05, "texture contrast does not rise" + what);
        expect(halo < 1e-5f, "a pixel overshoots its neighbourhood's range by more than a quarter (halo)" + what);
        expect(dark < 1e-5f, "the dark side of an edge falls below 3/4 of its darkest neighbour (goes to black)" + what);
        expect(colour < 1e-4f, "the colour's channel ratios change" + what);

        // Strength 3: more texture than at 1, overshoot within 0.75 of the range.
        const Pictures strong = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 3.0f, DlssNrMode_StructureHint);
        const double weaveStrong =
            spread(strong.painted, kSize / 2 + 3, kSize - 3) / spread(strong.proxy, kSize / 2 + 3, kSize - 3);
        float haloStrong = 0.0f, darkStrong = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
            {
                float lo = 1e30f, hi = -1e30f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const UINT xx = (UINT) std::clamp((int) x + dx, 0, (int) kSize - 1);
                        const UINT yy = (UINT) std::clamp((int) y + dy, 0, (int) kSize - 1);
                        const float l = Luma(strong.proxy.at(xx, yy));
                        lo = std::min(lo, l);
                        hi = std::max(hi, l);
                    }
                const float l = Luma(strong.painted.at(x, y));
                haloStrong = std::max({haloStrong, l - (hi + 0.75f * (hi - lo)), (lo - 0.75f * (hi - lo)) - l});
                darkStrong = std::max(darkStrong, 0.25f * lo - l);
            }
        std::printf("%-12s strength 3: weave %.0f%%, overshoot past 0.75 of the range %.2e\n", c.name, weaveStrong * 100.0,
                    haloStrong);
        expect(weaveStrong > weave + 0.2, "strength 3 does not lift texture beyond strength 1" + what);
        expect(haloStrong < 1e-5f, "strength 3 overshoots by more than 0.75 of the range" + what);
        expect(darkStrong < 1e-5f, "strength 3 takes the dark side of an edge below 1/4 of its darkest neighbour" + what);
    }

    // Smooth canvas: flat areas and the hard edge kept, texture flattened (more than the watercolour's grain), strength
    // 0 exact.
    auto spreadOf = [](const Image& im, UINT x0, UINT x1, UINT y0, UINT y1)
    {
        double s = 0.0, s2 = 0.0;
        int n = 0;
        for (UINT y = y0; y < y1; ++y)
            for (UINT x = x0; x < x1; ++x)
            {
                const double l = Luma(im.at(x, y));
                s += l;
                s2 += l * l;
                ++n;
            }
        const double m = s / n;
        return std::sqrt(std::max(s2 / n - m * m, 0.0));
    };

    // Two steps in the scene: the hard edge 0.03 | 0.9 at x = 32, and a smaller one 0.25 | 0.03 at x = 16 (about 3 EV,
    // a strong lip line). The 11x11 window reaches both from different pixels, so they are measured apart: the hard edge must
    // stay exact, the small step may soften a little (the brightness tolerance is wide on purpose, to flatten skin).
    std::printf("\n%-12s %10s %10s %10s %10s %10s %10s\n", "curve", "flat", "edge", "step", "grain", "weave", "off");
    for (const Case& c : cases)
    {
        const Pictures full = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 1.0f, DlssNrMode_SmoothCanvas);
        const Pictures none = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 0.0f, DlssNrMode_SmoothCanvas);

        float flat = 0.0f, edge = 0.0f, step = 0.0f, off = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
            {
                const Pixel& a = full.proxy.at(x, y);
                const Pixel& b = full.painted.at(x, y);
                const float d = std::max({std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
                if (x >= 3 && x < 11)
                    flat = std::max(flat, d);
                if (x >= 22 && x < 39) // both sides of the hard edge, out of reach of the small step and the texture
                    edge = std::max(edge, d);
                if (x >= 11 && x < 22) // both sides of the small step
                    step = std::max(step, d);
                const Pixel& n = none.painted.at(x, y);
                off = std::max({off, std::abs(n.r - a.r), std::abs(n.g - a.g), std::abs(n.b - a.b)});
            }

        const double grain =
            spreadOf(full.painted, 50, kSize - 6, 6, kSize / 2 - 6) / spreadOf(full.proxy, 50, kSize - 6, 6, kSize / 2 - 6);
        const double weave = spreadOf(full.painted, 50, kSize - 6, kSize / 2 + 6, kSize - 6) /
                             spreadOf(full.proxy, 50, kSize - 6, kSize / 2 + 6, kSize - 6);

        std::printf("%-12s %10.2e %10.2e %10.2e %9.0f%% %9.0f%% %10.2e\n", c.name, flat, edge, step, grain * 100.0,
                    weave * 100.0, off);
        const std::string what = std::string(" (smooth canvas, ") + c.name + ")";
        expect(flat < 1e-4f, "flat areas change" + what);
        expect(edge < 1e-3f, "the hard edge moves or blurs" + what);
        expect(step < 0.03f, "the small step softens by more than 0.03" + what);
        expect(grain < 0.3 && weave < 0.3, "texture is not flattened to under 30%" + what);
        expect(off == 0.0f, "strength 0 changes the picture" + what);
    }

    // Smooth canvas, what a brightness-only bilateral got wrong (tone-mapped frame and two HDR curves):
    //  - colour edge: skin | lip red of the SAME luminance must stay apart (it was smeared);
    //  - specks: lone pixels 1 EV darker than flat skin flatten to under a quarter of their depth (they kept half);
    //  - thin line: a 1-pixel line keeps over 90% of its depth (a line is not a speck);
    //  - catchlight: a lone pixel 1.7 EV brighter (an eye's catchlight) keeps over 90% of its height;
    //  - reach: the grain painted twice as large on a 2160-line picture is flattened like the plain grain on 1080
    //    lines (within 5 points).
    std::printf("\n%-12s %10s %10s %10s %10s %10s\n", "curve", "colour", "speck", "line", "grain1080", "grain2160");
    for (const Case& c : cases)
    {
        if (!(c.passthrough != 0 || c.reversible == 5 || c.reversible == 6))
            continue;
        const std::string what = std::string(" (smooth canvas, ") + c.name + ")";

        // Colour edge at x = 32, both sides luminance 0.30.
        const Pixel skin {0.85f, 0.62f, 0.50f, 1}, lip {0.80f, 0.40f, 0.42f, 1};
        const float skinScale = 0.30f / Luma(skin), lipScale = 0.30f / Luma(lip);
        Image colour {kSize, kSize, std::vector<Pixel>(kSize * kSize)};
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
                colour.at(x, y) = x < 32 ? Pixel {skin.r * skinScale, skin.g * skinScale, skin.b * skinScale, 1}
                                         : Pixel {lip.r * lipScale, lip.g * lipScale, lip.b * lipScale, 1};
        const Pictures colourOut = Paint(gpu, nr.Get(), colour, c.passthrough, c.reversible, 1.0f, DlssNrMode_SmoothCanvas);
        float colourMove = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 26; x < 38; ++x)
            {
                const Pixel& a = colourOut.proxy.at(x, y);
                const Pixel& b = colourOut.painted.at(x, y);
                colourMove = std::max({colourMove, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
            }

        // Specks (every 6th pixel both ways, 0.15 on 0.30) and a vertical 1-pixel line (0.03) at x = 52.
        Image specks {kSize, kSize, std::vector<Pixel>(kSize * kSize)};
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
            {
                float v = 0.30f;
                if (x < 44 && x % 6 == 3 && y % 6 == 3)
                    v = 0.15f;
                if (x == 52)
                    v = 0.03f;
                if (x == 58 && y == 32)
                    v = 1.0f; // the catchlight: display white, 1.7 EV above 0.30 (a tone-mapped picture clips above it)
                specks.at(x, y) = {v, v, v, 1};
            }
        const Pictures speckOut = Paint(gpu, nr.Get(), specks, c.passthrough, c.reversible, 1.0f, DlssNrMode_SmoothCanvas);
        double speck = 0.0, line = 1.0;
        const double catchlight = (Luma(speckOut.painted.at(58, 32)) - Luma(speckOut.painted.at(60, 32))) /
                                  (Luma(speckOut.proxy.at(58, 32)) - Luma(speckOut.proxy.at(60, 32)));
        int speckCount = 0;
        for (UINT y = 6; y < kSize - 6; ++y)
            for (UINT x = 0; x < 44; ++x)
                if (x % 6 == 3 && y % 6 == 3 && x >= 6)
                {
                    const double flat = Luma(speckOut.proxy.at(x - 3, y)); // a background pixel between specks
                    const double before = flat - Luma(speckOut.proxy.at(x, y));
                    const double after = Luma(speckOut.painted.at(x - 3, y)) - Luma(speckOut.painted.at(x, y));
                    speck = std::max(speck, after / before);
                    ++speckCount;
                }
        for (UINT y = 6; y < kSize - 6; ++y)
        {
            const double before = Luma(speckOut.proxy.at(56, y)) - Luma(speckOut.proxy.at(52, y));
            const double after = Luma(speckOut.painted.at(56, y)) - Luma(speckOut.painted.at(52, y));
            line = std::min(line, after / before);
        }

        // Reach: grain on 64 x 1080 vs the same grain at twice the size on 128 x 2160.
        auto grainImage = [](UINT w, UINT h, UINT scale)
        {
            Image im {w, h, std::vector<Pixel>(w * h)};
            for (UINT y = 0; y < h; ++y)
                for (UINT x = 0; x < w; ++x)
                {
                    uint32_t seed = (x / scale) * 73856093u ^ (y / scale) * 19349663u;
                    seed = seed * 1664525u + 1013904223u;
                    seed ^= seed >> 13;
                    seed *= 1274126177u;
                    const float grain = ((seed >> 8) & 0xFFFF) / 32767.5f - 1.0f;
                    const float v = 0.3f * (1.0f + 0.2f * grain);
                    im.at(x, y) = {v, v, v, 1};
                }
            return im;
        };
        const Pictures g1 = Paint(gpu, nr.Get(), grainImage(64, 1080, 1), c.passthrough, c.reversible, 1.0f,
                                  DlssNrMode_SmoothCanvas);
        const Pictures g2 = Paint(gpu, nr.Get(), grainImage(128, 2160, 2), c.passthrough, c.reversible, 1.0f,
                                  DlssNrMode_SmoothCanvas);
        const double grain1 = spreadOf(g1.painted, 12, 52, 500, 580) / spreadOf(g1.proxy, 12, 52, 500, 580);
        const double grain2 = spreadOf(g2.painted, 24, 104, 1000, 1160) / spreadOf(g2.proxy, 24, 104, 1000, 1160);

        std::printf("%-12s %10.2e %9.0f%% %9.0f%% %9.0f%% %9.0f%%\n", c.name, colourMove, speck * 100.0, line * 100.0,
                    grain1 * 100.0, grain2 * 100.0);
        expect(colourMove < 0.01f, "an edge of equal brightness but another colour is smeared" + what);
        expect(speckCount > 0 && speck < 0.25, "lone specks keep a quarter or more of their depth" + what);
        std::printf("%-12s %9.0f%% catchlight\n", "", catchlight * 100.0);
        expect(line > 0.9, "a thin line loses more than a tenth of its depth" + what);
        expect(catchlight > 0.9, "a catchlight loses more than a tenth of its height" + what);
        expect(std::abs(grain1 - grain2) < 0.05, "the same grain twice as large on twice the lines flattens differently" + what);
    }

    // Smooth canvas at strength 3 = three runs, each on the one before's result (DlssNrInput::Runs): the grain falls
    // further than after one run; the hard edge and the colour edge of equal brightness still stay.
    auto canvasRuns = [&](const Image& picture, uint32_t passthrough, uint32_t reversible, int runs)
    {
        Pictures first = Paint(gpu, nr.Get(), picture, passthrough, reversible, 1.0f, DlssNrMode_SmoothCanvas);
        Image current = first.painted;
        for (int r = 1; r < runs; ++r)
        {
            auto in = gpu.Texture(current.w, current.h, &current);
            auto out = gpu.Texture(current.w, current.h, nullptr);
            auto spare = gpu.Texture(current.w, current.h, nullptr);
            DlssNrConstants k = Base(DlssNrMode_SmoothCanvas, current.w, current.h, passthrough, reversible);
            gpu.Run(nr.Get(), k, in.Get(), out.Get(), spare.Get(), current.w, current.h);
            current = gpu.Read(out.Get());
        }
        return Pictures {first.proxy, current};
    };
    std::printf("\n%-12s %10s %10s %10s %10s\n", "curve", "grain1", "grain3", "edge3", "colour3");
    for (const Case& c : cases)
    {
        const std::string what = std::string(" (smooth canvas x3, ") + c.name + ")";
        const Pictures one = Paint(gpu, nr.Get(), frame, c.passthrough, c.reversible, 1.0f, DlssNrMode_SmoothCanvas);
        const Pictures three = canvasRuns(frame, c.passthrough, c.reversible, 3);
        float edge3 = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 22; x < 39; ++x)
            {
                const Pixel& a = three.proxy.at(x, y);
                const Pixel& b = three.painted.at(x, y);
                edge3 = std::max({edge3, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
            }
        const double grain1 =
            spreadOf(one.painted, 50, kSize - 6, 6, kSize / 2 - 6) / spreadOf(one.proxy, 50, kSize - 6, 6, kSize / 2 - 6);
        const double grain3 = spreadOf(three.painted, 50, kSize - 6, 6, kSize / 2 - 6) /
                              spreadOf(three.proxy, 50, kSize - 6, 6, kSize / 2 - 6);

        const Pixel skin {0.85f, 0.62f, 0.50f, 1}, lip {0.80f, 0.40f, 0.42f, 1};
        const float skinScale = 0.30f / Luma(skin), lipScale = 0.30f / Luma(lip);
        Image colour {kSize, kSize, std::vector<Pixel>(kSize * kSize)};
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
                colour.at(x, y) = x < 32 ? Pixel {skin.r * skinScale, skin.g * skinScale, skin.b * skinScale, 1}
                                         : Pixel {lip.r * lipScale, lip.g * lipScale, lip.b * lipScale, 1};
        const Pictures colourOut = canvasRuns(colour, c.passthrough, c.reversible, 3);
        float colour3 = 0.0f;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 26; x < 38; ++x)
            {
                const Pixel& a = colourOut.proxy.at(x, y);
                const Pixel& b = colourOut.painted.at(x, y);
                colour3 = std::max({colour3, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
            }

        std::printf("%-12s %9.0f%% %9.0f%% %10.2e %10.2e\n", c.name, grain1 * 100.0, grain3 * 100.0, edge3, colour3);
        expect(grain3 < grain1 * 0.75, "three runs do not flatten the grain to under 3/4 of one run's" + what);
        expect(edge3 < 1e-3f, "three runs move the hard edge" + what);
        expect(colour3 < 0.01f, "three runs smear an edge of equal brightness but another colour" + what);
    }

    // The tiled smooth canvas (D3D12's own build: each 16x16 group decodes its tile once into group-shared memory) gives
    // the reference's numbers: the scene, the colour edge, specks and line, grain on 2160 lines (the reach at its
    // widest, 10 pixels: the tile's border), a size that is no multiple of 16, faces only, strength 0.5.
    {
        auto tiledNr = gpu.Compile(argv[1], true);
        auto compare = [&](const Image& picture, uint32_t passthrough, uint32_t reversible, float strength, bool facesOnly)
        {
            const UINT w = picture.w, h = picture.h;
            auto source = gpu.Texture(w, h, &picture);
            auto proxy = gpu.Texture(w, h, nullptr);
            auto spare = gpu.Texture(w, h, nullptr);
            auto a = gpu.Texture(w, h, nullptr);
            auto b = gpu.Texture(w, h, nullptr);
            gpu.Run(nr.Get(), Base(DlssNrMode_Encode, w, h, passthrough, reversible), source.Get(), proxy.Get(), spare.Get(), w, h);
            DlssNrConstants k = Base(DlssNrMode_SmoothCanvas, w, h, passthrough, reversible);
            k.TransferStrength = strength;
            k.SkinProtection = facesOnly ? 1u : 0u;
            gpu.Run(nr.Get(), k, proxy.Get(), a.Get(), spare.Get(), w, h);
            gpu.Run(tiledNr.Get(), k, proxy.Get(), b.Get(), spare.Get(), w, h, 16);
            const Image ra = gpu.Read(a.Get()), rb = gpu.Read(b.Get());
            float most = 0.0f;
            for (size_t i = 0; i < ra.px.size(); ++i)
                most = std::max({most, std::abs(ra.px[i].r - rb.px[i].r), std::abs(ra.px[i].g - rb.px[i].g),
                                 std::abs(ra.px[i].b - rb.px[i].b), std::abs(ra.px[i].a - rb.px[i].a)});
            return most;
        };

        Image odd {70, 50, std::vector<Pixel>(70 * 50)};
        Image tall {96, 2160, std::vector<Pixel>(96 * 2160)};
        uint32_t seed = 7;
        for (Image* im : {&odd, &tall})
            for (UINT y = 0; y < im->h; ++y)
                for (UINT x = 0; x < im->w; ++x)
                {
                    seed = seed * 1664525u + 1013904223u;
                    const float g = ((seed >> 8) & 0xFFFF) / 32767.5f - 1.0f;
                    const float v = (x / 9 + y / 13) % 3 == 0 ? 0.05f : 0.3f * (1.0f + 0.25f * g);
                    im->at(x, y) = {v * 1.25f, v, v * 0.8f, 1};
                }

        std::printf("\n%-12s %10s\n", "curve", "tiled diff");
        for (const Case& c : cases)
        {
            float most = 0.0f;
            most = std::max(most, compare(frame, c.passthrough, c.reversible, 1.0f, false));
            most = std::max(most, compare(odd, c.passthrough, c.reversible, 1.0f, false));
            most = std::max(most, compare(odd, c.passthrough, c.reversible, 0.5f, true));
            most = std::max(most, compare(tall, c.passthrough, c.reversible, 1.0f, false));
            std::printf("%-12s %10.2e\n", c.name, most);
            expect(most == 0.0f, std::string("the tiled smooth canvas differs from the reference (") + c.name + ")");
        }
    }

    // Faces only: grey texture is left exactly as it was on every curve (no skin colour); skin-coloured texture is
    // smoothed (tone-mapped frame, where the stored values are the display colours the skin test is made for).
    std::printf("\n%-12s %10s %10s\n", "curve", "grey off", "skin");
    for (const Case& c : cases)
    {
        Image mixed = frame;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
                if (y >= kSize / 2) // the lower half's texture skin-coloured (a warm 0.85 : 0.62 : 0.50 tone)
                {
                    Pixel& p = mixed.at(x, y);
                    p = {p.r * 2.6f * 0.85f, p.g * 2.6f * 0.62f, p.b * 2.6f * 0.50f, 1};
                }

        float greyOff = 0.0f;
        double skin = 0.0;
        for (uint32_t mode : {DlssNrMode_Watercolor, DlssNrMode_StructureHint, DlssNrMode_SmoothCanvas})
        {
            const Pictures faces = Paint(gpu, nr.Get(), mixed, c.passthrough, c.reversible, 1.0f, mode, true);
            for (UINT y = 0; y < kSize / 2; ++y)
                for (UINT x = 0; x < kSize; ++x)
                {
                    const Pixel& a = faces.proxy.at(x, y);
                    const Pixel& b = faces.painted.at(x, y);
                    greyOff = std::max({greyOff, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
                }
            if (mode == DlssNrMode_SmoothCanvas)
                skin = spreadOf(faces.painted, 50, kSize - 6, kSize / 2 + 6, kSize - 6) /
                       spreadOf(faces.proxy, 50, kSize - 6, kSize / 2 + 6, kSize - 6);
        }

        std::printf("%-12s %10.2e %9.0f%%\n", c.name, greyOff, skin * 100.0);
        const std::string what = std::string(" (faces only, ") + c.name + ")";
        expect(greyOff == 0.0f, "a style changes grey (non-skin) texture" + what);
        if (c.passthrough != 0)
            expect(skin < 0.3, "skin-coloured texture is not smoothed" + what);
    }

    if (fails)
    {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::puts("PASS: watercolour keeps flat areas and hard edges and flattens texture; the structure hint raises texture\n"
              "      contrast with bounded halos and no colour change; the smooth canvas keeps flat areas and hard edges\n"
              "      and flattens texture; faces only leaves grey texture alone and smooths skin; strength 0 is a no-op\n"
              "      (WARP HLSL)");
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
