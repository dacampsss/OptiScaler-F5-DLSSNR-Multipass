// Closed-loop host check of F5Low on a native Vulkan swapchain: the game's interval measurement, our-wait accounting,
// and batch spacing (F5LowVkRules.h, the same functions F5Low_Vk.cpp calls) run against a
// virtual game and a virtual display, with each frame's measured interval and our waits fed back the way
// F5Low_Vk::QueuePresent does. No real sleeping. The "legacy" model (interval from raw arrival deltas, CPU spacing under
// FIFO) is the pre-fix behaviour and must run away, which proves the loop can fail.
//
// Part 2 (full loop, with f5low's real pacer core) needs the f5low-fg sources, which are not in this repo:
//   cl /std:c++20 /EHsc /W4 /I<f5low-fg>/include /I<f5low-fg>/src tests/f5low_vk_pacing_loop_smoke.cpp
// Without them part 2 is skipped (the CI build has no such path); part 1 builds with
//   cl /std:c++20 /EHsc /W4 tests/f5low_vk_pacing_loop_smoke.cpp
#define NOMINMAX
#include "../OptiScaler/framegen/f5low/F5LowVkRules.h"

#if __has_include("pace_api.h")
#define HAVE_PACER 1
#include "pace_api.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

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

using namespace F5LowVk;

namespace
{
// The display: FIFO takes one image per vblank and a present blocks while the swapchain's images are all queued;
// IMMEDIATE never blocks (each present is shown at once).
struct Display
{
    bool fifo = true;
    double vblankMs = 8.333;
    size_t images = 3; // set per run: the game's images plus one per generated frame
    std::deque<double> slots; // vblank time of each queued image
    double lastSlot = 0.0;

    // Returns when the present call returns
    double Present(double t)
    {
        if (!fifo)
            return t;

        while (!slots.empty() && slots.front() <= t)
            slots.pop_front();

        while (slots.size() >= images)
        {
            t = std::max(t, slots.front());
            slots.pop_front();
        }

        const double next = (std::floor(t / vblankMs) + 1.0) * vblankMs;
        lastSlot = std::max(lastSlot + vblankMs, next);
        slots.push_back(lastSlot);
        return t;
    }
};

struct Result
{
    std::vector<double> interval; // our smoothed interval after each game frame
    std::vector<double> period;   // time between two game presents
    double endMs = 0.0;
};

struct Cfg
{
    double workMs;
    double hz;
    int multiplier;
    bool fifo;
    bool legacy;
    int frames = 900;
};

Result Run(const Cfg& cfg)
{
    Display disp;
    disp.fifo = cfg.fifo;
    disp.vblankMs = 1000.0 / cfg.hz;

    const uint32_t n = GeneratedCount(cfg.multiplier - 1, kMaxGenerated, kMaxGenerated);
    // The swapchain has the game's 3 images plus one per frame we generate (RequestedMinImageCount)
    disp.images = RequestedMinImageCount(3, n, 0);

    Result r;
    double t = 0.0, interval = 0.0, lastEntry = 0.0, ourWait = 0.0;
    uint32_t seed = 12345;

    for (int f = 0; f < cfg.frames; ++f)
    {
        // The game's own CPU+GPU time, +-5 % deterministic jitter
        seed = seed * 1664525u + 1013904223u;
        const double jitter = (static_cast<double>(seed >> 16) / 65535.0 - 0.5) * 0.10;
        t += cfg.workMs * (1.0 + jitter);

        // QueuePresent entry
        const double entry = t;
        if (lastEntry > 0.0)
        {
            const double raw = entry - lastEntry;
            interval = NextInterval(interval, cfg.legacy ? raw : Unthrottled(raw, ourWait));
        }

        lastEntry = entry;

        // Generate: nothing until the interval is known
        const uint32_t gen = interval > 0.0 ? n : 0;
        const double sp = cfg.legacy ? SpacingMs(interval, gen) : BatchSpacingMs(interval, gen, cfg.fifo);
        const double base = t;

        for (uint32_t k = 0; k < gen; ++k)
        {
            if (k > 0 && sp > 0.0)
                t = std::max(t, base + k * sp);

            t = disp.Present(t);
        }

        if (gen > 0 && sp > 0.0)
            t = std::max(t, base + gen * sp);

        t = disp.Present(t); // the game's own frame; the game thread returns now

        ourWait = t - entry; // AddOurWait of everything this hook blocked on
        r.interval.push_back(interval);
        r.period.push_back(f > 0 ? entry - r.endMs : 0.0);
        r.endMs = entry;
    }

    return r;
}

double Mean(const std::vector<double>& v, int from, int to)
{
    double s = 0.0;

    for (int i = from; i < to; ++i)
        s += v[i];

    return s / (to - from);
}

bool Near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fabs(b); }
} // namespace


#ifdef HAVE_PACER
// ---------------------------------------------------------------------------------------------------------------------
// Part 2: the whole loop of the production flow, with f5low's real pacer core (pace_api.h).
//   Game thread (QueuePresent / AcquireNextImage): the latency gate (WaitForPresent of the present `bound` back, bound =
//     LatencyBound(game images 3, generated)), the game's acquire (a blocking image wait and the swapchain lock the worker
//     holds across its presents), the hook (slot / job-two-back waits), interval measurement from the arrival time minus
//     the waits that are ours, the real arrival ticks for the pacer.
//   Present worker (one job at a time): f5low's pacer plans the batch; a display-paced (FIFO) batch is not timed
//     (FixedPacing=display), a MAILBOX / IMMEDIATE one is; acquire an image, wait for the GPU, present under the
//     swapchain lock (a present into a full FIFO queue blocks inside it).
//   GPU: the game's frame is done kGpuTail after the present call (the ready stamp), f5low's generated frames kGenMs
//     after that (the done stamp).
//   Display: FIFO takes one image per vblank, IMMEDIATE shows at once.
// Everything on a virtual clock. Not modelled: flip metering and the burst (RTX 50 only), DirectPresent's pass-through.
// The "794c" model is the code at 794c9858f as it behaved (synthetic arrival ticks, worker pickup as the ready stamp,
// timed FIFO batches, gate bound 2): printed, it shows the RDR2 symptoms. "legacy" is before that commit.
// ---------------------------------------------------------------------------------------------------------------------

// The synthetic arrival tick of 794c9858f (no longer in F5LowVkRules.h), kept here only to model that build
static int64_t NextSyntheticTick794c(int64_t previous, int64_t real, double intervalMs, int64_t ticksPerSecond)
{
    if (previous == 0 || intervalMs <= 0.0 || ticksPerSecond <= 0)
        return real;

    const int64_t next = previous + static_cast<int64_t>(intervalMs * static_cast<double>(ticksPerSecond) / 1000.0);
    const int64_t drift = next > real ? next - real : real - next;
    return drift > ticksPerSecond / 5 ? real : next;
}

namespace loop
{
constexpr double kFreq = 10000000.0;
int64_t T(double ms) { return static_cast<int64_t>(ms * kFreq / 1000.0); }
double M(int64_t t) { return static_cast<double>(t) * 1000.0 / kFreq; }

struct Display
{
    bool fifo = true;
    double vblank = 8.333;
    size_t images = 3;
    struct Rec
    {
        double presentT, slot;
    };
    std::deque<Rec> recs;
    double lastSlot = 0.0;
    double lastShown = 0.0; // when the last present reaches the screen

    size_t Queued(double t) const
    {
        size_t c = 0;

        for (const auto& r : recs)
            if (r.presentT <= t && r.slot > t)
                ++c;

        return c;
    }

    double WaitBelow(double t, size_t limit)
    {
        if (!fifo)
            return t;

        while (!recs.empty() && recs.front().slot < t - 500.0)
            recs.pop_front();

        while (Queued(t) >= limit)
        {
            double next = 1e300;

            for (const auto& r : recs)
                if (r.presentT <= t && r.slot > t)
                    next = std::min(next, r.slot);

            t = next;
        }

        return t;
    }

    double Acquire(double t) { return WaitBelow(t, images - 1); } // one image is on screen, one is the one acquired

    double Present(double t)
    {
        if (!fifo)
        {
            lastShown = t;
            return t;
        }

        t = WaitBelow(t, images);
        const double next = (std::floor(t / vblank) + 1.0) * vblank;
        lastSlot = std::max(lastSlot + vblank, next);
        recs.push_back({ t, lastSlot });
        lastShown = lastSlot;
        return t;
    }
};

struct Model
{
    const char* name;
    bool unthrottled;    // interval = arrival delta minus our waits, else the raw delta (before 794c9858f)
    bool synthArrive;    // the pacer's arrival ticks are synthetic (794c9858f), else the real present time
    bool readyGpu;       // readyTicks = game GPU done (watcher), else the worker's pickup (794c9858f)
    bool legacySpace;    // plain spacing = SpacingMs under FIFO too (before 794c9858f)
    bool pacer;          // f5low's pacer plans the batch (default), else the plain spacing
    uint32_t gate;       // latency gate: 0 off, 255 = LatencyBound(3, n, display paced), else a fixed bound
    bool pacerKnowsGate; // the pacer is made with the gate flag (production); false = made without it (before the rework)
    bool countAcquire;   // the game's acquire block and lock polling are counted as our wait
    bool clampCpuInterval; // ClampRelease with the game's CPU-only interval (before the rework), else the pacer's own
    int switchAt;        // the gate goes to `switchGate` at this frame and the pacer is made again (0: never)
    bool switchGate;
};

struct Out
{
    std::vector<double> interval, pacerInterval, period, realShown, gateWait;
    std::vector<double> presents; // when each of our presents returned
    std::vector<size_t> presentsAt; // presents.size() after each game frame
    uint32_t releases = 0, clamped = 0; // timed waits after the warm-up, and how many ClampRelease moved earlier
    double endMs = 0.0;
    uint32_t n = 0;
};

Out Run(const Cfg& c, const Model& m)
{
    const uint32_t n = GeneratedCount(c.multiplier - 1, kMaxGenerated, kMaxGenerated);
    Display d;
    d.fifo = c.fifo;
    d.vblank = 1000.0 / c.hz;
    d.images = RequestedMinImageCount(3, n, 0);

    // f5low's pacer, made for a latency gate state as the export makes it (the host's gate flag, f5low's own setting on auto)
    const auto makePacer = [&](bool gateOn)
    {
        pacecore::Config pc;
        pc.ticksPerSecond = kFreq;
        pc.refreshHz = c.hz;
        pc.fixedDisplay =
            m.pacerKnowsGate && !pacesched::FixedTimer(pacesched::HostFixedMode(0, pacesched::kFixedAuto), gateOn);
        return std::make_unique<pacecore::Pacer>(pc);
    };
    bool gateOn = m.gate != 0;
    auto pacerPtr = makePacer(gateOn);

    constexpr double kGpuTail = 2.0, kGenMs = 1.5, kHookMs = 0.3, kPresentCpuMs = 0.25, kPollMs = 0.15;
    constexpr int64_t kTps = static_cast<int64_t>(kFreq);
    uint32_t bound = m.gate == 255 ? LatencyBound(3, n, c.fifo) : m.gate;

    Out o;
    o.n = n;
    std::vector<double> done;
    std::deque<std::pair<double, double>> lockHeld; // [start, end) of the worker's presents under the swapchain lock
    double tg = 0.0, wfree = 0.0, interval = 0.0, lastEntry = 0.0, ourWait = 0.0;
    int64_t synth = 0;
    uint32_t seed = 12345;

    // The game's acquire polls for the swapchain lock; a present under it that blocks in a full FIFO queue holds it
    const auto lockFree = [&](double t)
    {
        for (bool moved = true; moved;)
        {
            moved = false;

            for (const auto& iv : lockHeld)
                if (iv.first <= t && t < iv.second)
                {
                    t = iv.second + kPollMs;
                    moved = true;
                }
        }

        return t;
    };

    for (int f = 0; f < c.frames; ++f)
    {
        // A new swapchain with the other gate state: the gate follows and the pacer is made again for it
        if (m.switchAt != 0 && f == m.switchAt)
        {
            gateOn = m.switchGate;
            bound = gateOn ? LatencyBound(3, n, c.fifo) : 0;
            pacerPtr = makePacer(gateOn);
        }

        auto& pacer = *pacerPtr;

        // The latency gate (GateAcquire in the game's acquire): the present `bound` back is on screen
        double gateWaited = 0.0;

        if (bound > 0 && f >= static_cast<int>(bound))
        {
            const double until = o.realShown[f - bound];

            if (until > tg)
            {
                gateWaited = until - tg;
                tg = until;
            }
        }

        o.gateWait.push_back(gateWaited);

        const double acquireStart = tg;
        tg = lockFree(d.Acquire(tg));
        const double acquireWaited = m.countAcquire ? tg - acquireStart : 0.0;

        while (!lockHeld.empty() && lockHeld.front().second < tg - 200.0)
            lockHeld.pop_front();

        seed = seed * 1664525u + 1013904223u;
        tg += c.workMs * (1.0 + (static_cast<double>(seed >> 16) / 65535.0 - 0.5) * 0.10);

        const double entry = tg;
        const int64_t real = T(entry);

        if (lastEntry > 0.0)
        {
            const double raw = entry - lastEntry;
            interval = NextInterval(interval, m.unthrottled ? Unthrottled(raw, ourWait) : raw);
        }

        synth = NextSyntheticTick794c(synth, real, interval, kTps);
        const int64_t arrive = m.synthArrive ? synth : real;
        lastEntry = entry;

        // Generate: the job two frames back (the output set) and a free slot, our waits
        double ready = entry;

        if (f >= 2)
            ready = std::max(ready, done[f - 2]);

        if (f >= static_cast<int>(kSlots))
            ready = std::max(ready, done[f - kSlots]);

        tg = ready + kHookMs;
        ourWait = (ready - entry) + gateWaited + acquireWaited; // the hook's CPU work (kHookMs) is not a wait

        const double gpuDone = entry + kGpuTail, genDone = gpuDone + kGenMs;
        const uint32_t gen = interval > 0.0 ? n : 0;
        double t = std::max(wfree, tg);
        const double start = t;

        // One present under the swapchain lock: a full FIFO queue blocks inside the lock
        const auto presentLocked = [&](double at)
        {
            const double returned = d.Present(at);
            lockHeld.push_back({ at, returned + kPresentCpuMs });
            o.presents.push_back(returned);
            return returned + kPresentCpuMs;
        };

        if (m.pacer)
        {
            const double pickup = std::max(start, genDone);
            const int64_t readyTicks = gen == 0 ? 0 : (m.readyGpu ? T(gpuDone) : T(pickup));
            const int64_t doneTicks = m.readyGpu ? T(genDone) : T(pickup);

            F5LowPacerFrame fr {};
            fr.structSize = sizeof(fr);
            fr.arriveTicks = arrive;
            fr.readyTicks = readyTicks;
            fr.generated = gen;
            fr.displayPaced = c.fifo ? 1u : 0u;
            F5LowPacerBatch b {};
            b.structSize = sizeof(b);
            pacer.FrameArrived(fr, b);

            for (uint32_t k = 1; k <= gen + 1; ++k)
            {
                F5LowPacerTarget tgt {};
                tgt.structSize = sizeof(tgt);
                pacer.Target(b, k, tgt);

                if ((tgt.flags & F5LOW_PACER_TARGET_PACED) != 0 && tgt.releaseTicks != 0)
                {
                    const int64_t clampedTicks = ClampRelease(
                        tgt.releaseTicks, T(t), kTps, m.clampCpuInterval ? interval : PacerIntervalMs(b.stepTicks, gen, kTps));

                    if (f > 300)
                    {
                        ++o.releases;
                        o.clamped += clampedTicks < tgt.releaseTicks ? 1u : 0u;
                    }

                    t = std::max(t, M(clampedTicks));
                }

                if (k <= gen)
                {
                    t = d.Acquire(t);
                    t = std::max(t, genDone);
                }
                else
                    t = std::max(t, gpuDone);

                t = presentLocked(t);
                F5LowPacerPresented w {};
                w.structSize = sizeof(w);
                w.presentTicks = T(t);
                w.gpuDoneTicks = doneTicks;
                pacer.Presented(b, k, w);
            }

            pacer.BatchDone(b);
        }
        else
        {
            const double sp = m.legacySpace ? SpacingMs(interval, gen) : BatchSpacingMs(interval, gen, c.fifo);

            for (uint32_t k = 0; k < gen; ++k)
            {
                if (k > 0 && sp > 0.0)
                    t = std::max(t, start + k * sp);

                t = d.Acquire(t);
                t = std::max(t, genDone);
                t = presentLocked(t);
            }

            if (gen > 0 && sp > 0.0)
                t = std::max(t, start + gen * sp);

            t = presentLocked(std::max(t, gpuDone));
        }

        o.realShown.push_back(d.lastShown);
        o.presentsAt.push_back(o.presents.size());
        done.push_back(t);
        wfree = t;

        F5LowPacerInfo info {};
        pacer.Info(info);
        o.interval.push_back(interval);
        o.pacerInterval.push_back(info.intervalMs);
        o.period.push_back(f > 0 ? entry - o.endMs : 0.0);
        o.endMs = entry;
    }

    return o;
}
} // namespace loop

struct Shown
{
    double fps, spacingMs;
};

// Shown fps and the mean spacing between two shown frames over frames [a, b) of the run
static Shown ShownOver(const loop::Out& r, int a, int b)
{
    const double span = r.realShown[b - 1] - r.realShown[a - 1];
    const double frames = static_cast<double>(b - a) * (r.n + 1.0);
    return { span > 0.0 ? frames * 1000.0 / span : 0.0, span > 0.0 ? span / frames : 0.0 };
}

// The spacing of our presents over frames [a, b): mean, and the share of gaps shorter than a third of the mean (bunching)
struct Spacing
{
    double mean, bunched;
};

static Spacing PresentSpacing(const loop::Out& r, int a, int b)
{
    const size_t from = r.presentsAt[a - 1], to = r.presentsAt[b - 1];
    double sum = 0.0;
    size_t count = 0;

    for (size_t i = from + 1; i < to; ++i)
    {
        sum += r.presents[i] - r.presents[i - 1];
        ++count;
    }

    const double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
    size_t bunched = 0;

    for (size_t i = from + 1; i < to; ++i)
        if (r.presents[i] - r.presents[i - 1] < mean / 3.0)
            ++bunched;

    return { mean, count > 0 ? static_cast<double>(bunched) / static_cast<double>(count) : 0.0 };
}

// The host's FixedPacing rule and the pacer's answer must be the same for every gate state, present mode and setting
static void RuleMatchesPacer()
{
    int checked = 0;

    for (const bool gate : { false, true })
        for (int mode = 0; mode <= 3; ++mode)
            for (int setting = 0; setting <= 2; ++setting)
            {
                pacecore::Config pc;
                pc.ticksPerSecond = 1e7;
                const unsigned char own = setting == 1   ? pacesched::kFixedDisplay
                                          : setting == 2 ? pacesched::kFixedTimer
                                                         : pacesched::kFixedAuto;
                pc.fixedDisplay = !pacesched::FixedTimer(pacesched::HostFixedMode(0, own), gate);
                pacecore::Pacer pacer(pc);

                F5LowPacerFrame fr {};
                fr.structSize = sizeof(fr);
                fr.arriveTicks = fr.readyTicks = 1000000;
                fr.generated = 2;
                fr.displayPaced = VsyncPaces(mode) ? 1u : 0u;
                F5LowPacerBatch b {};
                b.structSize = sizeof(b);
                pacer.FrameArrived(fr, b);
                F5LowPacerTarget tgt {};
                tgt.structSize = sizeof(tgt);
                pacer.Target(b, 1, tgt);
                const bool timed = (tgt.flags & F5LOW_PACER_TARGET_PACED) != 0;
                CHECK(timed == !DisplaySpaces(gate, mode, setting));
                ++checked;
            }

    std::printf("\nFixedPacing rule: the host's DisplaySpaces equals the pacer's answer in %d of %d combinations "
                "(gate x present mode x setting)\n", checked, checked);
}

// A new swapchain with the other latency gate state in the middle of the run: the gate follows and the pacer is made again for it
// (F5Low_Vk::CreateSwapchain). Before and after the change the loop must be as flat as without it.
static void RecreateLoop()
{
    using loop::Model;
    const Model models[] = {
        { "recreate: gate on -> off", true, false, true, false, true, 255, true, true, false, 1500, false },
        { "recreate: gate off -> on", true, false, true, false, true, 0, true, true, false, 1500, true },
    };

    std::printf("\nGATE CHANGE ON A NEW SWAPCHAIN (frame 1500): shown fps before (1200-1300) and after (2800-2900), expect min(Hz, ...)\n");

    for (const Model& m : models)
    {
        int bad = 0, total = 0;

        for (const double hz : { 60.0, 120.0, 144.0 })
            for (const double w : { 5.0, 16.7, 25.0, 33.0 })
                for (int mult = 2; mult <= 6; ++mult)
                {
                    Cfg c { w, hz, mult, true, false };
                    c.frames = 3000;
                    const loop::Out r = loop::Run(c, m);
                    const double expGame = std::min(1000.0 / w, hz / mult);
                    const double expShown = std::min(hz, expGame * mult);
                    const Shown before = ShownOver(r, 1200, 1300), after = ShownOver(r, 2800, 2900);
                    const double gameBefore = 1000.0 / Mean(r.period, 1200, 1300), gameAfter = 1000.0 / Mean(r.period, 2800, 2900);
                    const bool ok = Near(before.fps, expShown, 0.10) && Near(after.fps, expShown, 0.10) &&
                                    Near(gameBefore, expGame, 0.10) && Near(gameAfter, expGame, 0.10);
                    ++total;
                    bad += ok ? 0 : 1;

                    if (!ok || (hz == 120.0 && (w == 5.0 || w == 25.0) && (mult == 2 || mult == 4)))
                        std::printf("  %-26s | %5.1f %3.0fF x%d | game %5.1f -> %5.1f (expect %5.1f) | shown %6.1f -> %6.1f (expect %6.1f)%s\n",
                                    m.name, w, hz, mult, gameBefore, gameAfter, expGame, before.fps, after.fps, expShown,
                                    ok ? "" : "  <-- off");
                }

        std::printf("  == %-26s: %d of %d cases outside the band\n", m.name, bad, total);
        CHECK(bad == 0);
    }
}

// A timed batch (MAILBOX / IMMEDIATE, or FIFO without the gate) whose generated frames finish after their target: the frame reports how
// late its own GPU work held it and the pacer lengthens the lead. The report needs f5low's done stamp; the production rule
// (DoneStampWaitMs) waits for it when the batch has a lead, the first version took what was there (0 for exactly these frames).
static void LeadLoop()
{
    constexpr double kFreqHz = 1e7;
    const auto ticks = [&](double ms) { return static_cast<int64_t>(ms * kFreqHz / 1000.0); };

    double leadWith = 0.0, leadWithout = 0.0;

    for (const bool production : { true, false })
    {
        pacecore::Config pc;
        pc.ticksPerSecond = kFreqHz;
        pacecore::Pacer pacer(pc);
        const uint32_t n = 2;
        double t = 100.0;

        for (int f = 0; f < 400; ++f)
        {
            t += 16.7;
            F5LowPacerFrame fr {};
            fr.structSize = sizeof(fr);
            fr.arriveTicks = fr.readyTicks = ticks(t);
            fr.generated = n;
            fr.displayPaced = 0;
            F5LowPacerBatch b {};
            b.structSize = sizeof(b);
            pacer.FrameArrived(fr, b);

            for (uint32_t k = 1; k <= n + 1; ++k)
            {
                F5LowPacerTarget tgt {};
                tgt.structSize = sizeof(tgt);
                pacer.Target(b, k, tgt);

                // f5low's work for a generated frame ends 4 ms after the frame's time; the frame goes out when it does
                const int64_t finished = tgt.targetTicks + ticks(4.0);
                F5LowPacerPresented w {};
                w.structSize = sizeof(w);
                w.presentTicks = k <= n ? finished : tgt.targetTicks;
                const bool stamped = production && DoneStampWaitMs(b.leadTicks, k, n) > 0;
                w.gpuDoneTicks = k <= n && stamped ? finished : 0;
                pacer.Presented(b, k, w);
            }

            pacer.BatchDone(b);
        }

        F5LowPacerInfo info {};
        pacer.Info(info);
        (production ? leadWith : leadWithout) = info.leadMs;
    }

    std::printf("\nLEAD ON A TIMED BATCH (generated frames 4 ms late, n=2, 60 fps): lead %.2f ms with the done stamp waited for, "
                "%.2f ms taking what is there (the first version)\n", leadWith, leadWithout);
    CHECK(leadWith > 4.0);       // the lead followed the late frames (3 ms start, up to the step)
    CHECK(leadWithout < 3.5);    // without the stamp it never moved: the check is able to fail
}

static void FullLoop()
{
    using loop::Model;
    RuleMatchesPacer();
    //  name, unthrottled, synthArrive, readyGpu, legacySpace, pacer, gate, pacerKnowsGate, countAcquire, clampCpuInterval, switchAt, switchGate
    const Model all[] = {
        { "legacy (before 794c), pacer", false, false, false, false, true, 2, false, false, true, 0, false },
        { "794c, pacer", true, true, false, false, true, 2, false, false, true, 0, false },
        { "794c, plain", true, true, false, false, false, 2, false, false, true, 0, false },
        { "new, pacer", true, false, true, false, true, 255, true, true, false, 0, false },
        { "new, plain", true, false, true, false, false, 255, true, true, false, 0, false },
        // one rule undone at a time (printed): each of them is needed
        { "new but pacer unaware of the gate", true, false, true, false, true, 255, false, true, false, 0, false },
        { "new but gate bound 2", true, false, true, false, true, 2, true, true, false, 0, false },
        { "new but pickup ready ticks", true, false, false, false, true, 255, true, true, false, 0, false },
        { "new but synthetic arrive", true, true, true, false, true, 255, true, true, false, 0, false },
        { "new but clamp by the CPU interval", true, false, true, false, true, 0, true, true, true, 0, false },
        // no latency gate: the timer paces a FIFO batch (FixedPacing auto without a gate, as D3D12)
        { "new, no gate, timer", true, false, true, false, true, 0, true, true, false, 0, false },
        { "new, no gate, plain", true, false, true, false, false, 0, true, true, false, 0, false },
    };

    const double works[] = { 5.0, 8.3, 16.7, 25.0, 33.0 };
    const double rates[] = { 60.0, 120.0, 144.0 };

    std::printf("\nFULL LOOP (3000 frames, gate on, latency bound for game images 3): per model, the cases outside the "
                "10%% band or with a drift; sampled rows at 120 Hz\n"
                "  columns: W ms, Hz F=FIFO I=IMMEDIATE, xN | game fps (expect) | shown fps (expect) | shown at frames "
                "300-400 -> 2800-2900 | present spacing ms (expect 1/Hz) | gate wait ms/frame | pacer interval ms\n");

    for (const Model& m : all)
    {
        int bad = 0, total = 0;
        double worstGame = 1.0, worstShown = 1.0, worstBunched = 0.0;

        for (const bool fifo : { true, false })
            for (const double hz : rates)
            {
                if (!fifo && hz != 120.0)
                    continue;

                for (const double w : works)
                    for (int mult = 2; mult <= 6; ++mult)
                    {
                        Cfg c { w, hz, mult, fifo, false };
                        c.frames = 3000;
                        const loop::Out r = loop::Run(c, m);
                        const double gameFps = 1000.0 / Mean(r.period, 2800, 2900);
                        const double expGame = fifo ? std::min(1000.0 / w, hz / mult) : 1000.0 / w;
                        const double expShown = fifo ? std::min(hz, expGame * mult) : expGame * mult;
                        const Shown a = ShownOver(r, 300, 400), b = ShownOver(r, 2800, 2900);
                        const double i1 = Mean(m.pacer ? r.pacerInterval : r.interval, 1500, 1600); // past the queue filling up
                        const double i2 = Mean(m.pacer ? r.pacerInterval : r.interval, 2800, 2900);
                        const double gw = Mean(r.gateWait, 2800, 2900);
                        // Where our timer spaces the batch (MAILBOX / IMMEDIATE, or FIFO without the gate) the presents are
                        // spread over it, not bunched (a release clamped by too small an interval bunches them)
                        const bool timedCase = !fifo || m.gate == 0;
                        const Spacing sp = PresentSpacing(r, 2800, 2900);
                        const double clampedShare = r.releases > 0 ? static_cast<double>(r.clamped) / r.releases : 0.0;
                        const bool spacingOk = !m.pacer || !timedCase || clampedShare <= 0.02;
                        const bool ok = Near(gameFps, expGame, 0.10) && Near(b.fps, expShown, 0.10) &&
                                        b.fps >= a.fps * 0.97 && spacingOk &&
                                        (m.pacer || i2 <= i1 * 1.05 + 0.2); // the pacer's own estimate is only printed
                        worstBunched = std::max(worstBunched, timedCase && m.pacer ? clampedShare : 0.0);
                        (void) sp;
                        ++total;
                        bad += ok ? 0 : 1;
                        worstGame = std::min(worstGame, gameFps / expGame);
                        worstShown = std::min(worstShown, b.fps / expShown);

                        const bool sample = hz == 120.0 && fifo && (mult == 2 || mult == 4 || mult == 6) &&
                                            (w == 5.0 || w == 16.7 || w == 25.0);

                        if (sample || (!ok && std::strncmp(m.name, "new, ", 5) == 0))
                            std::printf("  %-28s | %5.1f %3.0f%s x%d | %6.1f (%6.1f) | %6.1f (%6.1f) | %6.1f -> %6.1f | "
                                        "%5.2f (%5.2f) | %5.1f | %6.1f -> %6.1f%s\n",
                                        m.name, w, hz, fifo ? "F" : "I", mult, gameFps, expGame, b.fps, expShown,
                                        a.fps, b.fps, b.spacingMs, 1000.0 / hz, gw, i1, i2, ok ? "" : "  <-- off");
                    }
            }

        std::printf("  == %-34s: %3d of %3d cases outside the band, drifting or bunched; worst game fps %.2f, worst shown "
                    "fps %.2f of expected; worst share of releases moved earlier by the clamp (timed cases) %.2f\n",
                    m.name, bad, total, worstGame, worstShown, worstBunched);

        if (std::strncmp(m.name, "new, ", 5) == 0)
            CHECK(bad == 0); // the production flow holds everywhere, gate included
        else if (std::strncmp(m.name, "legacy", 6) == 0 || std::strstr(m.name, "clamp by the CPU") != nullptr)
            CHECK(bad > 0); // the loop is able to fail (the clamp one only through the spacing check)
    }
}
#endif

// A game that asks for MAILBOX (V-Sync off) with a cap above the refresh rate (RDR2: 25 fps at 5X = 125 images a second on a 120 Hz
// screen): MAILBOX shows the latest image at each refresh and replaces the others, IMMEDIATE shows every image (it may tear)
static void TearingLoop()
{
    const double hz = 120.0, vblankMs = 1000.0 / hz, spacingMs = 8.0; // 125 presents a second, evenly spaced (timer paced)
    const int presents = 1250;                                       // 10 s
    int shownMailbox = 0;
    long lastRefresh = -1;

    for (int i = 0; i < presents; ++i)
    {
        const long refresh = static_cast<long>(std::floor(i * spacingMs / vblankMs));

        if (refresh != lastRefresh)
            ++shownMailbox; // the first image since the last refresh; the next ones before the following refresh replace it

        lastRefresh = refresh;
    }

    int shownImmediate = 0;

    for (int i = 0; i < presents; ++i)
        ++shownImmediate; // every present is a flip

    const double droppedPerSecond = (presents - shownMailbox) / 10.0;
    std::printf("\nV-SYNC OFF, MAILBOX vs IMMEDIATE (125 images/s on 120 Hz): MAILBOX shows %d of %d images (%.1f replaced a second), IMMEDIATE "
                "shows %d (none replaced)\n", shownMailbox, presents, droppedPerSecond, shownImmediate);
    CHECK(droppedPerSecond > 3.0 && droppedPerSecond < 7.0); // about 5 a second
    CHECK(shownImmediate == presents);

    // the rule that picks IMMEDIATE for exactly this case
    CHECK(ChoosePresentMode(1, true, true, true, true).mode == 0);
}

int main()
{
    TearingLoop();

    const double works[] = { 8.3, 16.7, 25.0, 33.0 };
    const double rates[] = { 60.0, 120.0, 144.0 };

    std::printf("FIFO (fixed): W ms, Hz, xN | interval ms (own W) | period ms (expect max(W, N/Hz)) | game fps (expect) "
                "| shown fps (expect)\n");

    for (const double hz : rates)
        for (const double w : works)
            for (int mult = 2; mult <= 6; ++mult)
            {
                const Result r = Run({ w, hz, mult, true, false });
                const double n1 = mult;

                const double intFirst = Mean(r.interval, 300, 400), intLast = Mean(r.interval, 800, 900);
                const double period = Mean(r.period, 800, 900);
                const double expectPeriod = std::max(w, n1 * 1000.0 / hz);
                const double gameFps = 1000.0 / period, expectGame = std::min(1000.0 / w, hz / n1);
                const double shownFps = gameFps * n1, expectShown = std::min(hz, expectGame * n1);

                if (mult == 2 || mult == 4 || mult == 6)
                    std::printf("  %5.1f %5.0f x%d | %6.2f (%5.2f) | %6.2f (%6.2f) | %6.1f (%6.1f) | %6.1f (%6.1f)\n", w,
                                hz, mult, intLast, w, period, expectPeriod, gameFps, expectGame, shownFps, expectShown);

                CHECK(Near(intLast, w, 0.10)); // the game's own time, not inflated by our waits
                CHECK(intLast <= intFirst * 1.05 + 0.2); // no upward trend
                CHECK(Near(period, expectPeriod, 0.10)); // the display or the game limits, nothing else
                CHECK(Near(gameFps, expectGame, 0.10));
                CHECK(Near(shownFps, expectShown, 0.10));
                CHECK(intLast < 100.0);
            }

    std::printf("IMMEDIATE (fixed, our CPU spacing is real): W ms, xN | interval | period (expect W+n/(n+1)*(W-0.5)) "
                "| shown fps (expect)\n");

    for (const double w : works)
        for (int mult = 2; mult <= 6; ++mult)
        {
            const Result r = Run({ w, 120.0, mult, false, false });
            const double n = mult - 1.0;

            const double intFirst = Mean(r.interval, 300, 400), intLast = Mean(r.interval, 800, 900);
            const double period = Mean(r.period, 800, 900);
            const double expectPeriod = w + n / (n + 1.0) * (w - 0.5);
            const double shownFps = 1000.0 / period * (n + 1.0), expectShown = 1000.0 / expectPeriod * (n + 1.0);

            if (mult == 2 || mult == 4 || mult == 6)
                std::printf("  %5.1f x%d | %6.2f | %6.2f (%6.2f) | %6.1f (%6.1f)\n", w, mult, intLast, period,
                            expectPeriod, shownFps, expectShown);

            CHECK(Near(intLast, w, 0.10));
            CHECK(intLast <= intFirst * 1.05 + 0.2);
            CHECK(Near(period, expectPeriod, 0.10));
            CHECK(Near(shownFps, expectShown, 0.10));
            // a batch never outlasts the interval
            CHECK(n * BatchSpacingMs(intLast, static_cast<uint32_t>(n), false) < intLast);
        }

    // The pre-fix model must run away (so the loop above is able to fail): raw arrival deltas include our waits and the
    // spacing is applied under FIFO too
    std::printf("LEGACY (pre-fix, must run away): W ms, mode, xN | interval first 100 -> last 100 frames\n");

    for (const double w : works)
        for (const bool fifo : { true, false })
            for (int mult = 2; mult <= 6; ++mult)
            {
                const Result r = Run({ w, 120.0, mult, fifo, true });
                const double first = Mean(r.interval, 10, 110), last = Mean(r.interval, 800, 900);

                if (mult == 2 || mult == 4 || mult == 6)
                    std::printf("  %5.1f %s x%d | %6.2f -> %6.2f (own %5.1f)\n", w, fifo ? "FIFO " : "IMMED", mult,
                                first, last, w);

                CHECK(last > w * 1.3); // far above the game's own time
                CHECK(last > first * 1.15 || last > w * 1.8); // growing, or settled well above it
            }

    // The reported 33 -> 72 ms case: a 30 fps game at x2 on 120 Hz FIFO, legacy
    {
        const Result r = Run({ 33.0, 120.0, 2, true, true });
        std::printf("RDR2-like legacy (33 ms, 120 Hz FIFO, 2X): interval %.1f -> %.1f ms\n", Mean(r.interval, 10, 60),
                    Mean(r.interval, 800, 900));
        CHECK(Mean(r.interval, 800, 900) > 50.0);
    }

#ifdef HAVE_PACER
    FullLoop();
    RecreateLoop();
    LeadLoop();
#else
    std::printf("full loop skipped (no f5low-fg include path)\n");
#endif

    std::printf(fails == 0 ? "OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
