// Host check of misc/GpuMemTally.h: per-module buckets, frees, host-visible apart from GPU memory, and when a
// "GPU memory by module" line is due (>= 16 MB moved and >= 30 s since the last one), plus the final line.
// cl /std:c++20 /EHsc /W4 tests/gpu_mem_tally_smoke.cpp
#include "../OptiScaler/misc/GpuMemTally.h"

#include <cstdio>

using namespace gpumem;

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

static bool Has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

static int64_t MB(int64_t n) { return n * (int64_t) kMegabyte; }

int main()
{
    // heap classification
    CHECK(IsGpuMemory(HeapDefault, PoolUnknown));
    CHECK(!IsGpuMemory(HeapUpload, PoolUnknown));
    CHECK(!IsGpuMemory(HeapReadback, PoolUnknown));
    CHECK(!IsGpuMemory(HeapGpuUpload, PoolL1));
    CHECK(IsGpuMemory(HeapCustom, PoolL1));
    CHECK(!IsGpuMemory(HeapCustom, PoolL0));
    CHECK(AlignUp(1, 65536) == 65536);
    CHECK(AlignUp(65536, 65536) == 65536);
    CHECK(AlignUp(65537, 65536) == 131072);

    {
        Tally t;
        // first allocation of 300 MB: moved >= 16 MB and nothing logged yet -> a line right away
        std::string line = t.Add("f5low.dll", false, MB(300), 1000);
        CHECK(Has(line, "f5low.dll 300 MB"));
        CHECK(Has(line, "total 300 MB"));
        CHECK(Has(line, "host-visible 0 MB"));

        // a small change soon after: no line
        CHECK(t.Add("game", false, MB(5), 2000).empty());
        // big change but inside 30 s: no line
        CHECK(t.Add("game", false, MB(100), 20000).empty());
        // big change after 30 s: line with both buckets, sorted by size
        line = t.Add("nvngx_dlssnr.dll", false, MB(1210), 40000);
        CHECK(Has(line, "nvngx_dlssnr.dll 1210 MB"));
        CHECK(line.find("nvngx_dlssnr.dll") < line.find("f5low.dll"));
        CHECK(Has(line, "game 105 MB"));

        // free: bucket goes down, host-visible is kept apart
        CHECK(t.Add("f5low.dll", false, -MB(300), 41000).empty());
        CHECK(t.Gpu("f5low.dll") == 0);
        CHECK(t.Add("OptiScaler", true, MB(64), 42000).empty());
        CHECK(t.Host("OptiScaler") == (uint64_t) MB(64));
        CHECK(t.Gpu("OptiScaler") == 0);
        CHECK(t.TotalGpuBytes() == (uint64_t) MB(1210 + 105));

        // nothing moved since the last line but time passed: no line
        std::string quiet = t.Due(100000);
        CHECK(!quiet.empty()); // 300 MB freed + 64 MB host since the last line
        CHECK(t.Due(200000).empty());

        // a free of something that was never counted does not go below zero
        t.Add("other", false, -MB(10), 300000);
        CHECK(t.Gpu("other") == 0);

        // shutdown line: always, with the peak (1315 MB + 300 MB earlier at once = 1615 at most)
        std::string fin = t.Final();
        CHECK(Has(fin, "(final)"));
        CHECK(Has(fin, "peak total"));
        CHECK(Has(fin, "host-visible 64 MB"));
    }

    {
        // small buckets are folded into "others"
        Tally t;
        t.Add("a.dll", false, MB(20), 0);
        std::string line = t.Add("tiny.dll", false, 4096, 100000);
        CHECK(line.empty()); // too little moved
        line = t.Final();
        CHECK(Has(line, "a.dll 20 MB"));
        CHECK(!Has(line, "tiny.dll"));
        CHECK(Has(line, "others"));
    }

    // scopes
    CHECK(ModuleFileName(L"J:\\Games\\Hookshot\\bin\\OptiScaler\\plugins\\f5low.dll") == "f5low.dll");
    CHECK(ModuleFileName(L"C:/x/nvngx_dlssnr.dll") == "nvngx_dlssnr.dll");
    CHECK(ModuleFileName(L"libxess_fg.dll") == "libxess_fg.dll");
    CHECK(ProtectedUnder(ProtectFg, ModuleFileName(L"D:\\a\\f5low.dll")));
    CHECK(ParseProtectScope("auto") == -1); // auto = the default (fg), left to Config
    CHECK(ParseProtectScope("off") == ProtectOff);
    CHECK(ParseProtectScope("False") == ProtectOff);
    CHECK(ParseProtectScope("fg") == ProtectFg);
    CHECK(ParseProtectScope("FG") == ProtectFg);
    CHECK(ParseProtectScope("all") == ProtectAll);
    CHECK(ParseProtectScope("true") == ProtectAll);
    CHECK(ParseProtectScope("banana") == -1);
    CHECK(ProtectedUnder(ProtectFg, "f5low.dll"));
    CHECK(!ProtectedUnder(ProtectFg, "OptiScaler"));
    CHECK(!ProtectedUnder(ProtectFg, "nvngx_dlssnr.dll"));
    CHECK(ProtectedUnder(ProtectAll, "OptiScaler"));
    CHECK(ProtectedUnder(ProtectAll, "nvngx_dlssg.dll"));
    CHECK(!ProtectedUnder(ProtectAll, "libxess_fg.dll"));
    CHECK(!ProtectedUnder(ProtectOff, "f5low.dll"));
    CHECK(std::string(ProtectScopeName(ProtectFg)) == "fg");

    {
        // residency calls per module, by level, with known and unknown sizes
        Tally t;
        t.NoteResidency("libxess_fg.dll", 0xa0010000, (int64_t) MB(100));
        t.NoteResidency("libxess_fg.dll", 0xa0010000, (int64_t) MB(80));
        t.NoteResidency("sl.common.dll", 0x78000000, -1);
        t.NoteResidency("sl.common.dll", 0xa0010000, (int64_t) MB(10));
        std::string line = t.Final();
        CHECK(Has(line, "residency calls: libxess_fg.dll 2 objects HIGH 180 MB"));
        CHECK(Has(line, "sl.common.dll 2 objects NORMAL x1 HIGH x1 10 MB (+1 of unknown size)") ||
              Has(line, "sl.common.dll 2 objects HIGH x1 NORMAL x1 10 MB (+1 of unknown size)"));
        // a line is due because of them even though no memory moved
        Tally u;
        u.NoteResidency("x.dll", 0xa0010000, 0);
        CHECK(!u.Due(100000).empty());
        CHECK(u.Due(200000).empty());
        Tally w;
        CHECK(!Has(w.Final(), "residency"));
    }

    if (fails == 0)
        printf("PASS\n");
    return fails == 0 ? 0 : 1;
}
