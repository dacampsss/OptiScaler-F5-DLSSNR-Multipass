#include "pch.h"
#include "F5Low_Vk.h"

#include "F5Low.h"
#include "F5LowVkRules.h"
#include "f5low_frames.h"
#include "f5low_pacer.h"

#include <State.h>
#include <Util.h>
#include <low_latency/input/input_common.h>
#include <framegen/xefg/XeFG_DynamicTarget.h>
#include <Config.h>
#include <hooks/Vulkan_Hooks.h>
#include <menu/menu_common.h>
#include <native/NativeDriverVk.h>
#include <native/SharedFrameVk.h>
#include <misc/IdentifyGpu.h>
#include <dlssnr/DlssNr_VkExtensions.h>

#include <magic_enum.hpp>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>

using Microsoft::WRL::ComPtr;
using namespace F5LowVk;

namespace
{
constexpr VkImageUsageFlags kCopyUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr uint64_t kWaitFenceNs = 1'000'000'000ull;
constexpr double kAcquireMs = 100.0;
// VK_NV_present_metering's structure types are only in the headers' beta section
constexpr VkStructureType kStructSetPresentConfigNV = static_cast<VkStructureType>(1000613000);
constexpr VkStructureType kStructMeteringFeaturesNV = static_cast<VkStructureType>(1000613001);
constexpr DWORD kWorkerLockMs = kWorkerLockMsRule;
constexpr double kBlockMs = kBlockMsRule;

// The driver's swapchain calls, past our hooks (a hook would see them as the game's)
template <typename PFN> PFN SwapchainFunction(const char* name, PFN fallback)
{
    if (auto function = reinterpret_cast<PFN>(VulkanHooks::OriginalSwapchainFunction(name)))
        return function;

    return fallback;
}

// A per-thread high-resolution waitable timer (the worker and the game thread both wait)
HANDLE ThreadTimer()
{
    struct Holder
    {
        HANDLE handle =
            CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ~Holder()
        {
            if (handle != nullptr)
                CloseHandle(handle);
        }
    };
    thread_local Holder holder;
    return holder.handle;
}

// A short sleep that is not a 15 ms scheduler tick
void SleepHighRes(double ms)
{
    if (HANDLE timer = ThreadTimer(); timer != nullptr)
    {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(ms * 10000.0);

        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
        {
            WaitForSingleObject(timer, 50);
            return;
        }
    }

    std::this_thread::yield();
}

// A CPU wait until a time (Util::MillisecondsNow's clock): the timer for most of it, a spin for the rest
void WaitUntil(double targetMs)
{
    for (;;)
    {
        const double remaining = targetMs - Util::MillisecondsNow();

        if (remaining <= 0.0)
            return;

        if (remaining > 1.5)
            SleepHighRes(remaining - 1.0);
        else
            YieldProcessor();
    }
}

VkImageMemoryBarrier Barrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags src,
                             VkAccessFlags dst, uint32_t srcFamily = VK_QUEUE_FAMILY_IGNORED,
                             uint32_t dstFamily = VK_QUEUE_FAMILY_IGNORED)
{
    VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return barrier;
}

// A generated frame, as the D3D12 side left it, into an image the swapchain handed out; ready to present after.
void RecordOutput(VkCommandBuffer cmd, const native::SharedImageVk& shared, VkImage target, uint32_t family)
{
    VkImageMemoryBarrier before[2] = {
        Barrier(shared.Image(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0,
                VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_EXTERNAL, family),
        Barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT),
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, before);

    VkImageCopy region {};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.extent = { shared.Width(), shared.Height(), 1 };
    vkCmdCopyImage(cmd, shared.Image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier after[2] = {
        Barrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0),
        Barrier(shared.Image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_READ_BIT, 0, family, VK_QUEUE_FAMILY_EXTERNAL),
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, after);
}

bool UsableLayout(VkImageLayout layout)
{
    return layout != VK_IMAGE_LAYOUT_UNDEFINED && layout != VK_IMAGE_LAYOUT_PREINITIALIZED;
}

// Motion vector formats that cross as they are (a copy cannot convert); f5low converts the float ones itself
bool MotionFormat(VkFormat format, DXGI_FORMAT* dxgi)
{
    switch (format)
    {
    case VK_FORMAT_R16G16_SFLOAT:
        *dxgi = DXGI_FORMAT_R16G16_FLOAT;
        return true;
    case VK_FORMAT_R32G32_SFLOAT:
        *dxgi = DXGI_FORMAT_R32G32_FLOAT;
        return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        *dxgi = DXGI_FORMAT_R16G16B16A16_FLOAT;
        return true;
    default:
        return false;
    }
}

// magic_enum has no names for the depth/stencil formats (124-130)
std::string FormatName(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
        return "D16_UNORM";
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return "X8_D24_UNORM_PACK32";
    case VK_FORMAT_D32_SFLOAT:
        return "D32_SFLOAT";
    case VK_FORMAT_S8_UINT:
        return "S8_UINT";
    case VK_FORMAT_D16_UNORM_S8_UINT:
        return "D16_UNORM_S8_UINT";
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return "D24_UNORM_S8_UINT";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return "D32_SFLOAT_S8_UINT";
    default:
        return std::string(magic_enum::enum_name(format));
    }
}

constexpr uint32_t kBatches = 1 + kMaxGenerated; // the copy in, one copy out per generated frame

bool WaitFence12(ID3D12Fence* fence, uint64_t value, DWORD timeoutMs)
{
    if (fence == nullptr || fence->GetCompletedValue() >= value)
        return true;

    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    if (event == nullptr || FAILED(fence->SetEventOnCompletion(value, event)))
    {
        if (event != nullptr)
            CloseHandle(event);

        return false;
    }

    const bool done = WaitForSingleObject(event, timeoutMs) == WAIT_OBJECT_0;
    CloseHandle(event);
    return done;
}
} // namespace

// One game frame's generated frames, for whoever presents them
struct F5Low_Vk::Job
{
    uint32_t slot = 0;
    uint32_t generated = 0;
    uint64_t doneValue = 0;
    uint64_t readyValue = 0; // the value of the game-GPU-done fence this frame waits on (the watcher stamps it)
    uint32_t realIndex = 0;
    VkSemaphore realWait = VK_NULL_HANDLE;
    double intervalMs = 0.0;
    uint32_t set = 0;
    int64_t arriveTicks = 0;
    bool hasPresentId = false;
    uint64_t presentId = 0;
    bool hasFence = false;
    VkFence presentFence = VK_NULL_HANDLE;
    bool hasMode = false;
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    bool metered = false;  // the batch is spaced by the driver (VK_NV_present_metering), no CPU spacing
    bool hasGroup = false; // a single-device VkDeviceGroupPresentInfoKHR
    uint32_t groupCount = 0;
    uint32_t groupMask = 0;
    uint32_t groupMode = 0;
};

// The present mode a frame goes out with: the game's VkSwapchainPresentModeInfoEXT when it chained one, else the
// swapchain's
int F5Low_Vk::EffectiveMode(const Job& job) const
{
    return job.hasMode ? static_cast<int>(job.presentMode) : _presentMode;
}

struct F5Low_Vk::PacerCtx
{
    F5LowPacerBatch batch {};
    bool meterReported = false;
    int64_t readyTicks = 0; // when the game's frame was finished on the GPU, QPC (0: not seen)
};

struct F5Low_Vk::Impl
{
    // f5low.dll's frames-only API
    HMODULE module = nullptr;
    decltype(&f5lowFramesCreate) create = nullptr;
    decltype(&f5lowFramesBuildStatus) buildStatus = nullptr;
    decltype(&f5lowFramesGenerate) generate = nullptr;
    decltype(&f5lowFramesResize) resize = nullptr;
    decltype(&f5lowFramesDestroy) destroy = nullptr;
    decltype(&f5lowFramesLastError) lastError = nullptr;
    decltype(&f5lowFramesGpuSm) gpuSm = nullptr;

    // f5low's pacer: every export found, and the pacer made with the context
    decltype(&f5lowPacerCreate) pacerCreate = nullptr;
    decltype(&f5lowPacerDestroy) pacerDestroy = nullptr;
    decltype(&f5lowPacerFrameArrived) pacerFrameArrived = nullptr;
    decltype(&f5lowPacerTarget) pacerTarget = nullptr;
    decltype(&f5lowPacerStale) pacerStale = nullptr;
    decltype(&f5lowPacerMeterResult) pacerMeterResult = nullptr;
    decltype(&f5lowPacerPresented) pacerPresented = nullptr;
    decltype(&f5lowPacerBatchDone) pacerBatchDone = nullptr;
    decltype(&f5lowPacerGetInfo) pacerGetInfo = nullptr;
    decltype(&f5lowPacerNowTicks) pacerNowTicks = nullptr;
    decltype(&f5lowPacerTicksPerSecond) pacerTicksPerSecond = nullptr;
    decltype(&f5lowPacerWaitUntil) pacerWaitUntil = nullptr;
    decltype(&f5lowPacerTimerRaise) pacerTimerRaise = nullptr;
    bool pacerExports = false;
    F5LowPacer pacer = nullptr;
    int64_t ticksPerSecond = 0;

    F5LowFramesContext context = nullptr;
    DXGI_FORMAT contextFormat = DXGI_FORMAT_UNKNOWN;
    uint32_t contextFlags = 0;
    uint32_t contextWidth = 0, contextHeight = 0;
    bool d3dReady = false;

    // The private D3D12 device (the frame source's) and the queue and list of the depth stage
    native::VkInterop interop;
    ComPtr<ID3D12Device> device12;
    ComPtr<ID3D12CommandQueue> queue12;
    ComPtr<ID3D12CommandAllocator> allocators[kSlots];
    ComPtr<ID3D12GraphicsCommandList> list;

    // ready: Vulkan signals it when the copies in are done. depthDone: the D3D12 depth stage's own fence (one producer
    // each). consumed and done are f5low's. Every value only ever goes up.
    native::SharedFenceVk ready, consumed, done;
    ComPtr<ID3D12Fence> depthDone;
    FenceCounter depthCounter;
    uint64_t allocatorValue[kSlots] {};

    native::SharedImageVk color, hudless, motion, depthImage;
    native::SharedImageVk out[2][kMaxGenerated]; // two sets, written by f5low on alternate frames
    native::SharedBufferVk depthBuffer;
    ComPtr<ID3D12Resource> depthTexture;
    native::VkDepthCopyFormat depthFormat {};
    bool depthViaBuffer = false;
    VkFormat depthSource = VK_FORMAT_UNDEFINED;
    uint32_t depthWidth = 0, depthHeight = 0;

    // The swapchain's images, in the driver's order
    std::vector<VkImage> images;

    // Per frame in flight. The worker (or the game thread without one) uses the copy out parts of a slot while it is
    // busy; the game thread the rest.
    struct Slot
    {
        VkCommandBuffer commands[kBatches] {};
        VkFence fences[kBatches] {};
        bool used[kBatches] {};
        VkSemaphore inDone = VK_NULL_HANDLE;
        VkSemaphore acquired[kMaxGenerated] {};
        VkSemaphore outDone[kMaxGenerated] {};
    };
    VkCommandPool poolIn = VK_NULL_HANDLE;  // on the game's queue family
    VkCommandPool poolOut = VK_NULL_HANDLE; // on the present queue's family
    Slot slots[kSlots];
    uint32_t cursor = 0;
    uint32_t nextSet = 0;
    int setSlot[2] = { -1, -1 }; // the slot of the job that last used each set of output images
    bool slotsReady = false;
    bool deviceGone = false;
    bool warnedQueue = false;
    bool warnedNotReady = false;
    bool warnedPNext = false;

    // The present queue: our extra one when the device had a free queue, else the game's
    VkQueue outQueue = VK_NULL_HANDLE;
    uint32_t outFamily = UINT32_MAX;
    bool ownQueue = false;

    // The worker thread and its jobs. The swapchain is externally synchronised: swapchainMutex is held around every
    // acquire and present on it (the game's too), never across a wait.
    std::timed_mutex swapchainMutex;
    bool detached = false; // the worker was left behind at exit: this object is never destroyed
    std::set<int> unknownPNext;
    std::mutex jobMutex;
    std::condition_variable jobCv;
    std::deque<Job> jobs;
    bool busy = false;
    bool stop = false;
    bool slotBusy[kSlots] {};
    std::thread worker;
    std::atomic<int> lastResult { VK_SUCCESS };

    // The GPU stamps (D3D12's "ready stamp"): two watcher threads, one per fence kind, wait for the fences of each
    // frame and note the QPC when the game's frame was finished on the GPU (the ready fence) and when f5low's work was
    // (the done fence), whatever the worker is busy with. The pacer gets these, never the time the worker got to the
    // job. Each kind has its own thread, queue and fence, so a slow or never-signalled fence of one kind delays no
    // stamp of the other.
    struct WatchReq
    {
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0;
        uint64_t key = 0; // the StampBook key of the frame (StampKey)
        uint32_t slot = 0;
    };
    struct GraveEvent
    {
        HANDLE event = nullptr;
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0;
    };
    std::mutex watchMutex; // the queues, the stamps, the graveyard
    std::condition_variable watchCv;
    std::deque<WatchReq> watchQueue[2]; // indexed by StampBook::Kind
    std::thread watcher[2];
    HANDLE watchStopEvent = nullptr; // manual reset: set = the watchers leave their waits
    bool watchStop = false;
    StampBook stamps;
    // events a fence may still signal (a wait that timed out or was stopped): closed once the fence has passed the
    // value
    std::vector<GraveEvent> watchGraveyard;
    bool pacerGate = false; // the latency gate state the pacer was made for

    // Closes the events whose fence has passed the value they were set for (the signal has fired: nothing can hand it
    // to a reused handle); watchMutex held by the caller
    void SweepGraveyard()
    {
        std::erase_if(watchGraveyard,
                      [](GraveEvent& grave)
                      {
                          if (grave.fence == nullptr || grave.fence->GetCompletedValue() < grave.value)
                              return false;

                          CloseHandle(grave.event);
                          return true;
                      });
    }

    // The stop event and the swept graveyard, once the watchers are joined (a release, the destructor)
    void CloseWatchHandles()
    {
        {
            std::lock_guard lock(watchMutex);
            SweepGraveyard();
        }

        if (watchStopEvent != nullptr)
        {
            CloseHandle(watchStopEvent);
            watchStopEvent = nullptr;
        }
    }

    ~Impl()
    {
        if (!watcher[0].joinable() && !watcher[1].joinable())
            CloseWatchHandles();
    }
};

namespace
{
bool WatchersRunning(F5Low_Vk::Impl& s) { return s.watcher[0].joinable() && s.watcher[1].joinable(); }

void WatchPush(F5Low_Vk::Impl& s, StampBook::Kind kind, ID3D12Fence* fence, uint64_t value, uint64_t key, uint32_t slot)
{
    if (!WatchersRunning(s) || fence == nullptr)
        return;

    F5Low_Vk::Impl::WatchReq req;
    req.fence = fence;
    req.value = value;
    req.key = key;
    req.slot = slot;

    {
        std::lock_guard lock(s.watchMutex);
        s.watchQueue[kind].push_back(std::move(req));
    }

    s.watchCv.notify_all();
}

// 0: the fence reached the value, 1: the time ran out, 2: the watchers were stopped. A stop ends the wait at once.
int WaitFenceOrStop(F5Low_Vk::Impl& s, ID3D12Fence* fence, uint64_t value, DWORD timeoutMs)
{
    if (fence->GetCompletedValue() >= value)
        return 0;

    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);

    if (event == nullptr)
        return 1;

    if (FAILED(fence->SetEventOnCompletion(value, event)))
    {
        CloseHandle(event);
        return 1;
    }

    const HANDLE handles[2] = { event, s.watchStopEvent };
    const DWORD waited = WaitForMultipleObjects(2, handles, FALSE, timeoutMs);

    if (waited == WAIT_OBJECT_0)
    {
        CloseHandle(event);
        return 0;
    }

    // The fence may still signal this event later: it is kept until the fence has passed the value (closing it before
    // would hand the signal to whatever reuses the handle)
    {
        std::lock_guard lock(s.watchMutex);
        s.SweepGraveyard();
        F5Low_Vk::Impl::GraveEvent grave;
        grave.event = event;
        grave.fence = fence;
        grave.value = value;
        s.watchGraveyard.push_back(std::move(grave));
    }

    return waited == WAIT_OBJECT_0 + 1 ? 2 : 1;
}

void WatchLoop(F5Low_Vk::Impl* sp, StampBook::Kind kind)
{
    auto& s = *sp;

    for (;;)
    {
        F5Low_Vk::Impl::WatchReq req;

        {
            std::unique_lock lock(s.watchMutex);
            s.watchCv.wait(lock, [&] { return s.watchStop || !s.watchQueue[kind].empty(); });

            if (s.watchStop)
                return;

            req = std::move(s.watchQueue[kind].front());
            s.watchQueue[kind].pop_front();
        }

        // A bounded wait: a value that never comes (a removed device) must not hold the stamps of the frames behind it
        const int waited = WaitFenceOrStop(s, req.fence.Get(), req.value, 250);

        if (waited == 2)
            return;

        if (waited != 0)
            continue;

        const int64_t ticks = s.pacerNowTicks != nullptr ? s.pacerNowTicks() : 0;

        {
            std::lock_guard lock(s.watchMutex);
            s.stamps.Put(kind, req.slot, req.key, ticks);
        }

        s.watchCv.notify_all();
    }
}

// The stamp of a frame, waiting up to timeoutMs (0: take what is there) for the watcher to note it; 0 when it is not
// there (the pacer then uses the arrival time of the frame alone, never a mix of the two clocks within one frame)
int64_t WatchTake(F5Low_Vk::Impl& s, StampBook::Kind kind, uint32_t slot, uint64_t value, DWORD timeoutMs)
{
    if (!WatchersRunning(s) || value == 0)
        return 0;

    std::unique_lock lock(s.watchMutex);

    if (timeoutMs != 0)
        s.watchCv.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                           [&] { return s.stamps.Take(kind, slot, value) != 0; });

    return s.stamps.Take(kind, slot, value);
}
} // namespace

F5Low_Vk::F5Low_Vk() : IFGFeature_Vk(), IFGFeature(), _impl(std::make_unique<Impl>())
{
    _maxInterpolationCount = F5Low::MaxInterpolations(F5LOW_FRAMES_MAX_GENERATED,
                                                      Config::Instance()->FGXeFGMaxInterpolations.value_or_default());
    _nameCache = "F5Low";
    _generationOff = true; // until a frame really was generated
    UpdateCount();

    if (F5Low::LatchedGenerator() != F5Low::Generator::XefgTensor)
    {
        _blocked = "On the Vulkan native swapchain only XeFG Tensor is available";
        LOG_WARN("F5Low (Vulkan): {}", _blocked);
        return;
    }

    auto& s = *_impl;
    s.module = F5Low::LoadLibraryForOutput();

    if (s.module == nullptr)
    {
        _blocked = std::string("F5Low is not available: ") + F5Low::Available().reason;
        LOG_WARN("F5Low (Vulkan): {}", _blocked);
        return;
    }

    s.create = reinterpret_cast<decltype(s.create)>(GetProcAddress(s.module, "f5lowFramesCreate"));
    s.buildStatus = reinterpret_cast<decltype(s.buildStatus)>(GetProcAddress(s.module, "f5lowFramesBuildStatus"));
    s.generate = reinterpret_cast<decltype(s.generate)>(GetProcAddress(s.module, "f5lowFramesGenerate"));
    s.resize = reinterpret_cast<decltype(s.resize)>(GetProcAddress(s.module, "f5lowFramesResize"));
    s.destroy = reinterpret_cast<decltype(s.destroy)>(GetProcAddress(s.module, "f5lowFramesDestroy"));
    s.lastError = reinterpret_cast<decltype(s.lastError)>(GetProcAddress(s.module, "f5lowFramesLastError"));

    if (s.create == nullptr || s.buildStatus == nullptr || s.generate == nullptr || s.resize == nullptr ||
        s.destroy == nullptr || s.lastError == nullptr)
    {
        _blocked = "This f5low.dll has no frames-only API (f5lowFramesCreate and friends), it needs a newer f5low.dll";
        LOG_WARN("F5Low (Vulkan): {}", _blocked);
        s.module = nullptr;
        return;
    }

    // The presents of the generated frames run on a queue of their own (PrepareDevice reserves it with the device)
    _extraQueues.resize(1);
    _extraQueues[0].flags = VK_QUEUE_GRAPHICS_BIT;
    _extraQueues[0].present = true;

    s.gpuSm = reinterpret_cast<decltype(s.gpuSm)>(GetProcAddress(s.module, "f5lowFramesGpuSm"));

    const auto load = [&](auto& pointer, const char* name)
    {
        pointer = reinterpret_cast<std::remove_reference_t<decltype(pointer)>>(GetProcAddress(s.module, name));
        return pointer != nullptr;
    };

    s.pacerExports = load(s.pacerCreate, "f5lowPacerCreate") & load(s.pacerDestroy, "f5lowPacerDestroy") &
                     load(s.pacerFrameArrived, "f5lowPacerFrameArrived") & load(s.pacerTarget, "f5lowPacerTarget") &
                     load(s.pacerStale, "f5lowPacerStale") & load(s.pacerMeterResult, "f5lowPacerMeterResult") &
                     load(s.pacerPresented, "f5lowPacerPresented") & load(s.pacerBatchDone, "f5lowPacerBatchDone") &
                     load(s.pacerGetInfo, "f5lowPacerGetInfo") & load(s.pacerNowTicks, "f5lowPacerNowTicks") &
                     load(s.pacerTicksPerSecond, "f5lowPacerTicksPerSecond") &
                     load(s.pacerWaitUntil, "f5lowPacerWaitUntil") & load(s.pacerTimerRaise, "f5lowPacerTimerRaise");

    LOG_INFO("F5Low (Vulkan): frames-only API found, up to {} generated frames per game frame; f5low's pacer {}",
             _maxInterpolationCount,
             s.pacerExports ? "found" : "missing (an older f5low.dll): plain spacing, metering off unless forced");
}

F5Low_Vk::~F5Low_Vk()
{
    Shutdown();

    // A worker left behind at exit may still use everything in it
    if (_impl->detached)
        (void) _impl.release();
}

feature_version F5Low_Vk::Version()
{
    feature_version version;
    version.major = F5LOW_FRAMES_API_VERSION;
    return version;
}

const char* F5Low_Vk::Name() { return _nameCache.c_str(); }

void F5Low_Vk::SetReason(const std::string& reason)
{
    std::lock_guard lock(_reasonMutex);

    if (_reason != reason)
    {
        _reason = reason;

        if (!reason.empty())
            LOG_WARN("F5Low (Vulkan): {}", reason);
    }
}

std::string F5Low_Vk::StatusLine() const
{
    std::lock_guard lock(_reasonMutex);

    if (!_blocked.empty())
        return _blocked;

    return _reason;
}

void F5Low_Vk::UpdateCount(bool perFrame)
{
    auto* config = Config::Instance();
    const int target =
        config->FGInterpolationCount(config->FGXeFGInterpolationCount.value_or_default(), _maxInterpolationCount);
    const bool dynamic =
        DynamicRuns(config->FGXeFGDynamicMFG.value_or_default(), config->FGFollowsGame(), _maxInterpolationCount);

    if (dynamic && perFrame)
    {
        if (config->FGXeFGTargetAsCapDMFG.value_or_default() && !_warnedCapMode)
        {
            _warnedCapMode = true;
            LOG_INFO("F5Low (Vulkan): Dynamic MFG's \"use target as cap\" is not available here, the count follows the "
                     "target without holding it");
        }

        _dynamicActive = true;
        const int chosen = DynamicCount(target);

        if (chosen != _wanted)
        {
            LOG_INFO("F5Low (Vulkan): Dynamic MFG count {} -> {}", _wanted, chosen);
            _wanted = chosen;
        }
    }
    else if (!dynamic)
    {
        if (_dynamicActive)
        {
            _dynamicCount.Reset();
            _shownPerFrame.Rebase();
            _shownCount = -1;
            _dmfgHold = false;
            _dynamicKept = 0;
            _dynamicActive = false;
            _dynamicDrain = true;
        }

        if (target >= 1 && target != _wanted)
        {
            LOG_INFO("F5Low (Vulkan): interpolation count {} -> {}", _wanted, target);
            _wanted = target;
        }
    }

    // What the multiplier combos, the cap and the status read: the chosen count, as far as the generator and the spare
    // images allow it (what a frame really did is _generationOff and the status line)
    _framesToInterpolate = static_cast<int>(std::max(
        1u, GeneratedCount(_wanted, static_cast<uint32_t>(std::min(_maxInterpolationCount, _countLimit)), _headroom)));
}

// Dynamic MFG as on D3D12 (XeFG_Dx12::DynamicCount), count mode: the count follows the frame rate target, within the
// setting. "Use target as cap" is not ported. 1X (count 0) generates nothing but keeps feeding the generator.
int F5Low_Vk::DynamicCount(int maxCount)
{
    auto& state = State::Instance();
    auto* config = Config::Instance();
    const double now = Util::MillisecondsNow();

    const float refresh = DisplayRefreshFps(now);
    const float target =
        xefg::DynamicTargetFps(config->FGXeFGFramerateTargetDMFG.value_or_default(),
                               config->FramerateLimit.value_or_default(), refresh, VsyncPaces(_presentMode));

    const int kept = _dynamicKept >= 1 ? std::min(_dynamicKept, maxCount) : maxCount;
    const int current = _dmfgHold ? 0 : kept;

    // In the background the game is throttled: the count is kept, and it starts over once the game is back in front
    DWORD pid = 0;

    if (HWND front = GetForegroundWindow(); front != nullptr)
        GetWindowThreadProcessId(front, &pid);

    if (!InFront(pid, GetCurrentProcessId()))
    {
        if (!_dynamicUnfocused)
            LOG_INFO("F5Low (Vulkan): Dynamic MFG: the game's window lost focus, keeping {}X", current + 1);

        _dynamicUnfocused = true;
        _dynamicDrain = true;
        return CountAfterDynamic(current, kept);
    }

    if (_dynamicUnfocused)
    {
        LOG_INFO("F5Low (Vulkan): Dynamic MFG: the game's window is back in front, starting over at {}X", current + 1);
        _dynamicUnfocused = false;
        _dynamicCount.Reset();
        _shownPerFrame.Rebase();
        _shownCount = -1;
    }

    // Shown per real frame, measured: started over at a new count
    if (current != _shownCount)
    {
        _shownPerFrame.Restart();
        _shownCount = current;
    }

    _shownPerFrame.Update(_shownTotal.load());

    // What the game waited in OptiScaler (its low latency sleep, our gate and drain); piled up while Dynamic was off:
    // dropped
    double sleepMs = InputCommon::take_sleep_ms();
    double waitedMs = _gameWaitMs.exchange(0.0);

    if (_dynamicDrain)
    {
        sleepMs = 0.0;
        waitedMs = 0.0;
        _dynamicDrain = false;
    }

    const int rounding = std::clamp(config->FGXeFGRoundingDMFG.value_or_default(), 0, 2);
    const int minCount = config->FGXeFGDynamicAllow1X.value_or_default() || maxCount == 1 ? 0 : 1;

    const int count = _dynamicCount.Update({ now, state.lastFGFrameTime, sleepMs + waitedMs, target, refresh,
                                             config->FramerateLimit.value_or_default(), maxCount, current, rounding,
                                             minCount, _shownPerFrame.Value(), _shownPerFrame.Sample() });

    if (count != current || now - _dynamicLogMs >= 5000.0)
    {
        const auto line = std::format("F5Low (Vulkan): Dynamic MFG {}X{}: presented {:.1f} fps, target {:.1f}",
                                      current + 1, count != current ? std::format(" -> {}X", count + 1) : std::string(),
                                      _dynamicCount.PresentedFps(), target);

        if (count != current)
            LOG_INFO("{}", line);
        else
            LOG_DEBUG("{}", line);

        _dynamicLogMs = now;
    }

    _dmfgHold = DynamicHolds(true, count);

    if (count >= 1)
        _dynamicKept = count;

    return CountAfterDynamic(count, kept);
}

float F5Low_Vk::DisplayRefreshFps(double nowMs)
{
    // The window can move to another display: looked up again every 2 s
    if (_hwnd != nullptr && (_refreshCheckedMs <= 0.0 || nowMs - _refreshCheckedMs >= 2000.0))
    {
        _refreshCheckedMs = nowMs;
        _refreshRate = Util::GetActiveRefreshRate(_hwnd);
    }

    return static_cast<float>(std::max(_refreshRate, 0));
}

void F5Low_Vk::Activate()
{
    if (_isActive || !_blocked.empty() || _swapchain == VK_NULL_HANDLE)
        return;

    UpdateTarget();
    _isActive = true;
    LOG_DEBUG("");
}

void F5Low_Vk::Deactivate()
{
    if (_isActive)
        LOG_DEBUG("");

    _isActive = false;
}

void F5Low_Vk::EvaluateState(FG_Constants& fgConstants)
{
    OwnedLockGuard lock(Mutex, 555);

    auto& state = State::Instance();
    _constants = fgConstants;

    if (_swapchain == VK_NULL_HANDLE || state.isShuttingDown)
        return;

    if (!Config::Instance()->IsFGEnabled() || state.fgChanged)
        Deactivate();

    if (state.fgChanged)
    {
        state.fgChanged = false;
        UpdateTarget();
    }

    state.scChanged = false;
}

// ---- the worker and the swapchain lock ----

bool F5Low_Vk::DrainWorker(DWORD timeoutMs)
{
    auto& s = *_impl;

    if (!s.worker.joinable())
        return true;

    std::unique_lock lock(s.jobMutex);
    return s.jobCv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return s.jobs.empty() && !s.busy; });
}

// The worker idle and stopped, for whatever replaces or destroys what it uses. When it does not go idle in time it is
// left alone (running, with its state untouched) and the output latches off.
bool F5Low_Vk::QuiesceWorker(DWORD timeoutMs)
{
    if (!_impl->worker.joinable())
        return true;

    if (!DrainWorker(timeoutMs))
    {
        LatchOff("The present worker did not go idle in time, frame generation is off");
        return false;
    }

    StopWorker(true);
    return true;
}

void F5Low_Vk::LatchOff(const std::string& reason)
{
    _failures = 99;
    _generationOff = true;
    SetReason(reason);
}

void F5Low_Vk::StartWorker()
{
    auto& s = *_impl;

    if (!s.ownQueue || s.detached)
        return;

    // The GPU stamp watchers go with the worker, once there is a pacer to give the stamps to
    if (s.pacer != nullptr && s.pacerNowTicks != nullptr && !s.watcher[0].joinable() && !s.watcher[1].joinable())
    {
        if (s.watchStopEvent == nullptr)
            s.watchStopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);

        if (s.watchStopEvent != nullptr)
        {
            ResetEvent(s.watchStopEvent);

            {
                std::lock_guard lock(s.watchMutex);
                s.watchStop = false;
                s.watchQueue[0].clear();
                s.watchQueue[1].clear();
            }

            s.watcher[StampBook::Ready] = std::thread(WatchLoop, &s, StampBook::Ready);
            s.watcher[StampBook::Done] = std::thread(WatchLoop, &s, StampBook::Done);
        }
    }

    if (s.worker.joinable())
        return;
    if (s.worker.joinable())
        return;

    {
        std::lock_guard lock(s.jobMutex);
        s.stop = false;
    }

    s.worker = std::thread([this] { WorkerLoop(); });
}

// join: the caller drained it (QuiesceWorker); without: process exit, the thread is left behind and everything it may
// touch is leaked on purpose (the destructor)
void F5Low_Vk::StopWorker(bool join)
{
    auto& s = *_impl;

    if (s.watcher[0].joinable() || s.watcher[1].joinable())
    {
        {
            std::lock_guard lock(s.watchMutex);
            s.watchStop = true;
        }

        if (s.watchStopEvent != nullptr)
            SetEvent(s.watchStopEvent); // the waits end now: no 250 ms for the game thread to sit through

        s.watchCv.notify_all();

        for (auto& watcher : s.watcher)
        {
            if (!watcher.joinable())
                continue;

            if (join)
                watcher.join();
            else
            {
                watcher.detach();
                s.detached = true;
            }
        }
    }

    if (!s.worker.joinable())
        return;

    {
        std::lock_guard lock(s.jobMutex);
        s.stop = true;
    }

    s.jobCv.notify_all();

    if (!join)
    {
        s.worker.detach();
        s.detached = true;
        return;
    }

    s.worker.join();

    std::lock_guard lock(s.jobMutex);
    s.busy = false;

    for (auto& busy : s.slotBusy)
        busy = false;
}

void F5Low_Vk::WorkerLoop()
{
    auto& s = *_impl;

    for (;;)
    {
        Job job;

        {
            std::unique_lock lock(s.jobMutex);
            s.jobCv.wait(lock, [&] { return s.stop || !s.jobs.empty(); });

            // Jobs still queued when it is told to stop are presented first
            if (s.jobs.empty())
                return;

            job = s.jobs.front();
            s.jobs.pop_front();
            s.busy = true;
        }

        // f5low's pacer plans the batch (when to present each frame, whether the driver meters it); without it the
        // plain spacing and the fallback metering rule
        PacerCtx pacerCtx;
        PacerCtx* pacer = nullptr;

        if (s.pacer != nullptr)
        {
            // When the game's frame was finished on the GPU, and when f5low's work was: the watcher's stamps (bounded
            // wait, 0 when not seen). Never the time this thread got to the job.
            pacerCtx.readyTicks = WatchTake(s, StampBook::Ready, job.slot, job.readyValue, 30);

            F5LowPacerFrame frame {};
            frame.structSize = sizeof(frame);
            frame.arriveTicks = job.arriveTicks != 0 ? job.arriveTicks : s.pacerNowTicks();
            frame.generated = job.generated;
            frame.readyTicks = pacerCtx.readyTicks;
            frame.displayPaced = VsyncPaces(EffectiveMode(job)) ? 1u : 0u;
            pacerCtx.batch.structSize = sizeof(pacerCtx.batch);

            if (s.pacerFrameArrived(s.pacer, &frame, &pacerCtx.batch) == F5LOW_PACER_OK)
            {
                pacer = &pacerCtx;
                job.metered = MeterThisBatch(true, pacerCtx.batch.meterWanted, false, _hostMeter && !_meterHealth.off);
            }
        }

        const uint32_t presentedGenerated = RunGenerated(job, pacer == nullptr && PacedOnCpu(job.metered), pacer);

        if (pacer != nullptr)
            PacerStart(*pacer, job.generated + 1, false);

        // The game's own image after the generated ones, with what the game chained to its present
        VkPresentIdKHR presentId { VK_STRUCTURE_TYPE_PRESENT_ID_KHR };
        presentId.swapchainCount = 1;
        presentId.pPresentIds = &job.presentId;

        VkSwapchainPresentFenceInfoEXT fenceInfo { VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT };
        fenceInfo.swapchainCount = 1;
        fenceInfo.pFences = &job.presentFence;

        VkSwapchainPresentModeInfoEXT modeInfo { VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT };
        modeInfo.swapchainCount = 1;
        modeInfo.pPresentModes = &job.presentMode;

        VkDeviceGroupPresentInfoKHR groupInfo { VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR };
        groupInfo.swapchainCount = job.groupCount;
        groupInfo.pDeviceMasks = &job.groupMask;
        groupInfo.mode = static_cast<VkDeviceGroupPresentModeFlagBitsKHR>(job.groupMode);

        // The batch's size: presented frames + 1 (a cut-short batch is smaller); the first present after metered ones
        // that is not metered closes the run with 1. Chained on every present of the batch: whether the driver wants it
        // only on the first or on all of them is not confirmed (first run).
        const uint32_t batch = BatchConfig(job.metered, _meterWasOn.exchange(false), presentedGenerated);
        VkSetPresentConfigNV meterConfig { kStructSetPresentConfigNV };
        meterConfig.numFramesPerBatch = batch;

        const void* chain = nullptr;

        if (batch != 0)
        {
            meterConfig.pNext = chain;
            chain = &meterConfig;
        }

        if (job.hasGroup)
        {
            groupInfo.pNext = chain;
            chain = &groupInfo;
        }

        if (job.hasPresentId)
        {
            presentId.pNext = chain;
            chain = &presentId;
        }

        if (job.hasFence)
        {
            fenceInfo.pNext = chain;
            chain = &fenceInfo;
        }

        if (job.hasMode)
        {
            modeInfo.pNext = chain;
            chain = &modeInfo;
        }

        VkPresentInfoKHR real { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        real.pNext = chain;
        real.waitSemaphoreCount = 1;
        real.pWaitSemaphores = &job.realWait;
        real.swapchainCount = 1;
        real.pSwapchains = &_swapchain;
        real.pImageIndices = &job.realIndex;

        const auto realResult = PresentOn(s.outQueue, real, kWorkerLockMs);

        if (realResult == VK_TIMEOUT && pacer != nullptr)
            PacerPresented(*pacer, job, job.generated + 1, true, VK_SUCCESS, 0);

        if (const auto presented = realResult; presented != VK_TIMEOUT)
        {
            s.lastResult = presented;

            if (job.metered)
                NoteMeterFeedback(presented, meterConfig.presentConfigFeedback);

            if (pacer != nullptr)
                PacerPresented(*pacer, job, job.generated + 1, false, presented, meterConfig.presentConfigFeedback);
        }

        if (pacer != nullptr)
            s.pacerBatchDone(s.pacer, &pacer->batch);

        _meterWasOn = job.metered && !_meterHealth.off;
        xefg::AddShownPresent(_shownTotal, presentedGenerated + (realResult == VK_TIMEOUT ? 0u : 1u));

        {
            std::lock_guard lock(_statMutex);
            _statGenerated += presentedGenerated;
            ++_statBatches;
        }

        {
            std::lock_guard lock(s.jobMutex);
            s.busy = false;
            s.slotBusy[job.slot] = false;
        }

        s.jobCv.notify_all();
    }
}

// VK_TIMEOUT: the swapchain lock was not free in time (nothing was presented)
VkResult F5Low_Vk::PresentOn(VkQueue queue, const VkPresentInfoKHR& info, DWORD lockTimeoutMs)
{
    const auto presentFn = SwapchainFunction<PFN_vkQueuePresentKHR>("vkQueuePresentKHR", vkQueuePresentKHR);
    std::unique_lock lock(_impl->swapchainMutex, std::defer_lock);

    if (!lock.try_lock_for(std::chrono::milliseconds(lockTimeoutMs)))
    {
        LOG_WARN("F5Low (Vulkan): the swapchain lock was not free for {} ms, a present is dropped", lockTimeoutMs);
        return VK_TIMEOUT;
    }

    const auto result = presentFn(queue, &info);

    // Every image we put on the screen, for the stats line. The overlay's output rate is fed by the present function
    // itself (VulkanHooks' FGOutputQueuePresent counts each call), so it is not counted again here.
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
        ++_statShown;

    return result;
}

// The real present on the game's thread: after the worker's jobs (a short wait), so the order of the images holds.
// Never longer than the game thread's budget in all.
VkResult F5Low_Vk::DirectPresent(VkQueue queue, const VkPresentInfoKHR* info, uint64_t ownId, uint32_t meterBatch)
{
    const double startMs = Util::MillisecondsNow();

    // Our present id (the game chained none) and the metering batch go in front of the game's chain
    VkPresentInfoKHR copy = *info;
    VkPresentIdKHR idInfo { VK_STRUCTURE_TYPE_PRESENT_ID_KHR };
    VkSetPresentConfigNV meterInfo { kStructSetPresentConfigNV };

    // The first present after a metered run that is not metered closes it
    if (meterBatch == 0 && _meteringAvailable && _meterWasOn.exchange(false))
        meterBatch = 1;

    if (ownId != 0 || meterBatch != 0)
    {
        const void* chain = info->pNext;

        if (ownId != 0)
        {
            idInfo.swapchainCount = 1;
            idInfo.pPresentIds = &ownId;
            idInfo.pNext = chain;
            chain = &idInfo;
        }

        if (meterBatch != 0)
        {
            meterInfo.numFramesPerBatch = meterBatch;
            meterInfo.pNext = chain;
            chain = &meterInfo;
        }

        copy.pNext = chain;
        info = &copy;
    }

    // A game's per-present mode info saying MAILBOX goes out as IMMEDIATE too when our swapchain was made with it (the
    // generated frames do). The game's structs are const and may be shared: nothing of theirs is written. The chain up
    // to and including the mode info is rebuilt from local copies (the structs we know the size of; ours stay as they
    // are, the driver writes into them), the rest is pointed to as it is. An unknown struct in front leaves the present
    // alone.
    alignas(8) unsigned char chainStore[8][64] {};
    VkPresentModeKHR localMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    VkPresentInfoKHR modePresent {};

    if (_tearing)
    {
        const auto sizeOf = [](VkStructureType type) -> size_t
        {
            switch (type)
            {
            case VK_STRUCTURE_TYPE_PRESENT_ID_KHR:
                return sizeof(VkPresentIdKHR);
            case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT:
                return sizeof(VkSwapchainPresentFenceInfoEXT);
            case VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR:
                return sizeof(VkDeviceGroupPresentInfoKHR);
            default:
                return 0;
            }
        };

        VkBaseOutStructure* previous = nullptr;
        VkBaseOutStructure* first = nullptr;
        size_t used = 0;
        bool ok = true;
        const VkBaseInStructure* modeNode = nullptr;

        const auto append = [&](VkBaseOutStructure* node)
        {
            (previous != nullptr ? previous->pNext : first) = node;
            previous = node;
        };

        for (auto* node = static_cast<const VkBaseInStructure*>(info->pNext); node != nullptr && ok; node = node->pNext)
        {
            if (node->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT)
            {
                modeNode = node;
                break;
            }

            if (node == reinterpret_cast<const VkBaseInStructure*>(&idInfo) ||
                node == reinterpret_cast<const VkBaseInStructure*>(&meterInfo))
            {
                append(const_cast<VkBaseOutStructure*>(reinterpret_cast<const VkBaseOutStructure*>(node)));
                continue;
            }

            const size_t size = sizeOf(node->sType);

            if (size == 0 || size > sizeof(chainStore[0]) || used >= 8)
            {
                ok = false;
                break;
            }

            std::memcpy(chainStore[used], node, size);
            append(reinterpret_cast<VkBaseOutStructure*>(chainStore[used++]));
        }

        if (ok && modeNode != nullptr)
        {
            const auto* mode = reinterpret_cast<const VkSwapchainPresentModeInfoEXT*>(modeNode);

            if (mode->swapchainCount == 1 && mode->pPresentModes != nullptr &&
                mode->pPresentModes[0] == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                VkSwapchainPresentModeInfoEXT local = *mode;
                local.pPresentModes = &localMode;
                static_assert(sizeof(VkSwapchainPresentModeInfoEXT) <= sizeof(chainStore[0]));

                if (used < 8)
                {
                    std::memcpy(chainStore[used], &local, sizeof(local));
                    append(reinterpret_cast<VkBaseOutStructure*>(chainStore[used++]));
                    previous->pNext =
                        const_cast<VkBaseOutStructure*>(reinterpret_cast<const VkBaseOutStructure*>(modeNode->pNext));
                    modePresent = *info;
                    modePresent.pNext = first;
                    info = &modePresent;
                }
            }
        }
    }

    if (!DrainWorker(kBlockMs))
        LOG_DEBUG("F5Low (Vulkan): the worker was still busy, presenting the game's image anyway");

    const double drainedMs = Util::MillisecondsNow();
    AddAtomic(_gameWaitMs, drainedMs - startMs);
    AddOurWait(drainedMs - startMs);

    const auto result = PresentOn(queue, *info, RemainingMs(kBlockMs, drainedMs - startMs));

    // The swapchain lock and the driver's block (FIFO with a full queue) are the game thread's wait too
    AddOurWait(Util::MillisecondsNow() - drainedMs);

    if (meterBatch > 1)
        NoteMeterFeedback(result, meterInfo.presentConfigFeedback);

    if (result == VK_TIMEOUT)
    {
        // Something holds the swapchain: the frame is dropped rather than the game held, generation stops
        LatchOff("The swapchain lock was held too long (a stuck present), frame generation is off");
        return VK_SUCCESS;
    }

    return result;
}

// The swapchain is externally synchronised and the worker presents to it while the game acquires: an acquire never
// waits inside the lock (a game's blocking acquire would hold it while the images only come back through the worker's
// presents), it polls.
VkResult F5Low_Vk::AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                                    VkFence fence, uint32_t* pImageIndex)
{
    // A game that ignores SUBOPTIMAL is told its swapchain is out of date, once (nothing is acquired then)
    if (swapchain == _swapchain)
    {
        std::lock_guard askLock(_askMutex);

        if (OnAcquireRecreate(_ask, Util::MillisecondsNow()))
        {
            LOG_INFO("F5Low (Vulkan): the game ignored the suboptimal ask for {}x, answering its acquire with "
                     "OUT_OF_DATE",
                     _wanted + 1);
            return VK_ERROR_OUT_OF_DATE_KHR;
        }
    }

    // The game does not run further ahead of what is shown than it would without our spare images
    // (a polling game, timeout 0, is told not yet rather than held)
    if (_gateEnabled && swapchain == _swapchain && !GateAcquire(device, swapchain, timeout))
        return VK_NOT_READY;

    // The polling and the swapchain lock of the game's acquire are our wait (the worker holds the lock for presents)
    const double acquireStartMs = Util::MillisecondsNow();
    const auto acquired = AcquireLoop(device, swapchain, timeout, semaphore, fence, pImageIndex);

    if (swapchain == _swapchain)
        AddOurWait(Util::MillisecondsNow() - acquireStartMs);

    return acquired;
}

VkResult F5Low_Vk::AcquireLoop(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                               VkFence fence, uint32_t* pImageIndex)
{
    const auto acquireFn = SwapchainFunction<PFN_vkAcquireNextImageKHR>("vkAcquireNextImageKHR", vkAcquireNextImageKHR);
    const double startMs = Util::MillisecondsNow();

    for (;;)
    {
        VkResult result = VK_NOT_READY;

        {
            std::unique_lock lock(_impl->swapchainMutex, std::defer_lock);

            if (lock.try_lock_for(std::chrono::milliseconds(2)))
                result = acquireFn(device, swapchain, 0, semaphore, fence, pImageIndex);
        }

        if (result != VK_NOT_READY && result != VK_TIMEOUT)
            return result;

        if (timeout == 0)
            return VK_NOT_READY;

        if (timeout != UINT64_MAX && (Util::MillisecondsNow() - startMs) * 1e6 >= static_cast<double>(timeout))
            return VK_TIMEOUT;

        SleepHighRes(0.3);
    }
}

// vkWaitForPresentKHR on the swapchain the worker presents to: a poll under the lock, the wait outside it
VkResult F5Low_Vk::WaitForPresent(VkDevice device, VkSwapchainKHR swapchain, uint64_t presentId, uint64_t timeout)
{
    const auto waitFn = SwapchainFunction<PFN_vkWaitForPresentKHR>("vkWaitForPresentKHR", nullptr);

    if (waitFn == nullptr)
        return VK_SUCCESS;

    const double startMs = Util::MillisecondsNow();

    for (;;)
    {
        VkResult result = VK_TIMEOUT;

        {
            std::unique_lock lock(_impl->swapchainMutex, std::defer_lock);

            if (lock.try_lock_for(std::chrono::milliseconds(2)))
                result = waitFn(device, swapchain, presentId, 0);
        }

        if (result != VK_TIMEOUT)
            return result;

        if (timeout != UINT64_MAX && (Util::MillisecondsNow() - startMs) * 1e6 >= static_cast<double>(timeout))
            return VK_TIMEOUT;

        SleepHighRes(0.5);
    }
}

bool F5Low_Vk::AcquireImage(VkSemaphore semaphore, uint32_t* index, double timeoutMs)
{
    const auto result =
        AcquireLoop(_device, _swapchain, static_cast<uint64_t>(timeoutMs * 1e6), semaphore, VK_NULL_HANDLE, index);
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

// What the game chained to its present that the real present of a generated frame carries over: the present id, the
// present fence and the present mode (one swapchain). Any other struct makes this frame pass through.
bool F5Low_Vk::CarryPresentStructs(const VkPresentInfoKHR* present, Job& job)
{
    for (auto* node = static_cast<const VkBaseInStructure*>(present->pNext); node != nullptr; node = node->pNext)
    {
        switch (node->sType)
        {
        case VK_STRUCTURE_TYPE_PRESENT_ID_KHR:
        {
            const auto* id = reinterpret_cast<const VkPresentIdKHR*>(node);

            if (id->swapchainCount >= 1 && id->pPresentIds != nullptr)
            {
                job.hasPresentId = true;
                job.presentId = id->pPresentIds[0];
            }
            break;
        }
        case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT:
        {
            const auto* fence = reinterpret_cast<const VkSwapchainPresentFenceInfoEXT*>(node);

            if (fence->swapchainCount >= 1 && fence->pFences != nullptr)
            {
                job.hasFence = true;
                job.presentFence = fence->pFences[0];
            }
            break;
        }
        case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT:
        {
            const auto* mode = reinterpret_cast<const VkSwapchainPresentModeInfoEXT*>(node);

            if (mode->swapchainCount >= 1 && mode->pPresentModes != nullptr)
            {
                job.hasMode = true;
                job.presentMode = static_cast<VkPresentModeKHR>(
                    ChainedPresentMode(_tearing, static_cast<int>(mode->pPresentModes[0])));
            }
            break;
        }
        case VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR:
        {
            const auto* group = reinterpret_cast<const VkDeviceGroupPresentInfoKHR*>(node);
            const uint32_t mask =
                group->swapchainCount >= 1 && group->pDeviceMasks != nullptr ? group->pDeviceMasks[0] : 0;

            // One device only: its mask goes to every present of the frame
            if (!DeviceGroupCarryable(group->swapchainCount, static_cast<uint32_t>(group->mode), mask))
            {
                if (_impl->unknownPNext.insert(static_cast<int>(node->sType)).second)
                    LOG_INFO("F5Low (Vulkan): the game's device group present info is not a single device (swapchains "
                             "{}, mode {}, mask {}), those frames pass through",
                             group->swapchainCount, static_cast<uint32_t>(group->mode), mask);
                return false;
            }

            job.hasGroup = true;
            job.groupCount = group->swapchainCount;
            job.groupMask = mask;
            job.groupMode = static_cast<uint32_t>(group->mode);
            break;
        }
        default:
            if (_impl->unknownPNext.insert(static_cast<int>(node->sType)).second)
                LOG_INFO("F5Low (Vulkan): the game chains a present struct of type {} that cannot be carried to the "
                         "real present of a generated frame, those frames pass through",
                         static_cast<int>(node->sType));
            return false;
        }
    }

    return true;
}

// ---- the swapchain ----

bool F5Low_Vk::CreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
                               const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    _retiredOld = false;

    if (!_blocked.empty())
        return false;

    if (device != _device || pCreateInfo == nullptr || pSwapchain == nullptr)
        return false;

    auto& s = *_impl;

    // The old swapchain and the surface are externally synchronised: nothing of ours presents to them while the driver
    // replaces them (this create, and the hook's own one if ours fails). The worker comes back with the next frame.
    if (!QuiesceWorker(2000))
        return false;

    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    VkFormat sharedFormat = VK_FORMAT_UNDEFINED;

    const auto support = ClassifySwapchain(static_cast<int>(pCreateInfo->imageFormat),
                                           static_cast<long long>(pCreateInfo->imageColorSpace));

    if (!SwapchainGenerates(support) || !native::VkSharedPictureFormat(pCreateInfo->imageFormat, &dxgi, &sharedFormat))
    {
        SetReason(std::format("F5Low: {} (format {}, colour space {})", SwapchainReason(support),
                              static_cast<int>(pCreateInfo->imageFormat),
                              static_cast<int>(pCreateInfo->imageColorSpace)));
        return false;
    }

    if ((pCreateInfo->imageUsage & kCopyUsage) != kCopyUsage)
    {
        SetReason("The surface does not allow copies of the swapchain's images");
        return false;
    }

    VkSurfaceCapabilitiesKHR caps {};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_physicalDevice, pCreateInfo->surface, &caps) != VK_SUCCESS)
        caps.maxImageCount = 0;

    const auto createFn = SwapchainFunction<PFN_vkCreateSwapchainKHR>("vkCreateSwapchainKHR", vkCreateSwapchainKHR);
    const auto imagesFn =
        SwapchainFunction<PFN_vkGetSwapchainImagesKHR>("vkGetSwapchainImagesKHR", vkGetSwapchainImagesKHR);

    UpdateCount();

    // Spare images for the frames we generate (one per generated frame of the multiplier). The game's pNext chain (the
    // latency mode among it) stays as it came.
    // With the latency bound the spare images cost the game no queue depth: ask for the most the session may generate
    // (no recreate for a multiplier change). Without it only what the multiplier needs, and a recreate for a raise.
    const bool gateWanted = _presentWaitOn && Config::Instance()->F5LowVulkanLatencyGate.value_or_default() &&
                            GateUsable(F5LowVk::LatencyBound(pCreateInfo->minImageCount, 1, false));

    VkSwapchainCreateInfoKHR info = *pCreateInfo;

    // V-Sync off (MAILBOX): IMMEDIATE instead, so every generated frame reaches the screen as on D3D12
    {
        bool immediateSupported = false;
        uint32_t modeCount = 0;

        if (vkGetPhysicalDeviceSurfacePresentModesKHR(_physicalDevice, pCreateInfo->surface, &modeCount, nullptr) ==
                VK_SUCCESS &&
            modeCount > 0)
        {
            std::vector<VkPresentModeKHR> modes(modeCount);

            if (vkGetPhysicalDeviceSurfacePresentModesKHR(_physicalDevice, pCreateInfo->surface, &modeCount,
                                                          modes.data()) == VK_SUCCESS)
            {
                for (uint32_t i = 0; i < modeCount; ++i)
                    immediateSupported = immediateSupported || modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
            }
        }

        // A list of the modes the swapchain may switch between (VK_EXT_swapchain_maintenance1) the game chained: ours
        // is made with the game's list, so IMMEDIATE has to be in it
        bool listAllows = true;

        for (auto* node = static_cast<const VkBaseInStructure*>(pCreateInfo->pNext); node != nullptr;
             node = node->pNext)
        {
            if (node->sType != VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT &&
                node->sType != VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT)
                continue;

            // The compatibility set of a swapchain is per present mode: a swapchain the game made for MAILBOX with its
            // own list (or scaling info) is not made for IMMEDIATE by us
            listAllows = false;
        }

        const auto chosen = ChoosePresentMode(static_cast<int>(pCreateInfo->presentMode), immediateSupported,
                                              listAllows, Config::Instance()->F5LowVulkanTearing.value_or_default(),
                                              Config::Instance()->IsFGEnabled());
        info.presentMode = static_cast<VkPresentModeKHR>(chosen.mode);
        _gamePresentMode = static_cast<int>(pCreateInfo->presentMode);
        _tearing = chosen.why == TearingWhy::Tearing;

        if (pCreateInfo->presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            LOG_INFO(
                "F5Low (Vulkan): the game asked for present mode {} (MAILBOX): {} ({}); chosen when the "
                "swapchain is made, frame generation switched on or off later takes effect with the next swapchain",
                _gamePresentMode, chosen.mode, TearingText(chosen.why));
    }

    info.minImageCount = RequestedMinImageCount(
        pCreateInfo->minImageCount, SpareImagesToAsk(gateWanted, _maxInterpolationCount, _wanted), caps.maxImageCount);
    _gameMinImages = pCreateInfo->minImageCount;

    VkResult result;

    {
        std::lock_guard lock(s.swapchainMutex);
        result = createFn(device, &info, pAllocator, pSwapchain);

        // A failed create retires the old swapchain (VUID-VkSwapchainCreateInfoKHR-oldSwapchain-01933): the second try,
        // and the hook's if this one fails, are made without it
        if (result != VK_SUCCESS && pCreateInfo->oldSwapchain != VK_NULL_HANDLE)
            _retiredOld = true;

        if (result != VK_SUCCESS && info.minImageCount != pCreateInfo->minImageCount)
        {
            LOG_WARN("F5Low (Vulkan): {} swapchain images were refused ({}), trying the game's own count",
                     info.minImageCount, magic_enum::enum_name(result));
            info.minImageCount = pCreateInfo->minImageCount;
            info.oldSwapchain = VK_NULL_HANDLE;
            result = createFn(device, &info, pAllocator, pSwapchain);
        }

        if (result != VK_SUCCESS && _tearing)
        {
            LOG_WARN("F5Low (Vulkan): IMMEDIATE was refused ({}), trying the game's MAILBOX",
                     magic_enum::enum_name(result));
            info.presentMode = pCreateInfo->presentMode;
            info.oldSwapchain = VK_NULL_HANDLE;
            _tearing = false;
            result = createFn(device, &info, pAllocator, pSwapchain);
        }
    }

    if (result != VK_SUCCESS)
    {
        LOG_ERROR("F5Low (Vulkan): vkCreateSwapchainKHR error: {}", magic_enum::enum_name(result));
        return false;
    }

    s.deviceGone = false;
    s.lastResult = VK_SUCCESS;
    uint32_t count = 0;
    imagesFn(device, *pSwapchain, &count, nullptr);
    s.images.assign(count, VK_NULL_HANDLE);

    if (count == 0 || imagesFn(device, *pSwapchain, &count, s.images.data()) != VK_SUCCESS)
        s.images.clear();

    _headroom = ImageHeadroom(static_cast<uint32_t>(s.images.size()), pCreateInfo->minImageCount);
    _swapchain = *pSwapchain;
    _swapchainFormat = pCreateInfo->imageFormat;
    _swapchainExtent = pCreateInfo->imageExtent;
    _colorSpace = pCreateInfo->imageColorSpace;
    _hdr10 = support == SwapchainSupport::OkHdr10;
    _hwnd = VulkanHooks::SurfaceWindow(pCreateInfo->surface);
    _hudlessMode = -1;
    _hudlessLogs = 0;
    _intervalMs = 0.0;
    _lastEntryMs = 0.0;
    _generationOff = true;
    _loggedBehind = false;
    _loggedGenerating = false;

    // The latency bound: present wait with present ids when the device has them (a fence per present through
    // VK_EXT_swapchain_maintenance1 would clash with fences the game chains itself)
    _presentMode = static_cast<int>(info.presentMode); // the mode our swapchain really has
    _gameMinImages = pCreateInfo->minImageCount;
    _gateCount = 1;
    _gateMode = _presentMode;
    _gateBound = F5LowVk::LatencyBound(pCreateInfo->minImageCount, 1, VsyncPaces(_presentMode));

    {
        std::lock_guard gateLock(_gateMutex);
        _issuedCount = 0;
        std::fill(std::begin(_issuedIds), std::end(_issuedIds), 0ull);
        _gate = {};
    }

    _meterWasOn = false;
    _gateEnabled = gateWanted && SwapchainFunction<PFN_vkWaitForPresentKHR>("vkWaitForPresentKHR", nullptr) != nullptr;
    LOG_INFO("F5Low (Vulkan): game asked for {} images, present mode {}{}, latency bound {} ({})",
             pCreateInfo->minImageCount, _presentMode,
             _tearing ? std::format(" (game asked {}: tearing like D3D12)", _gamePresentMode) : std::string(),
             _gateBound,
             _gateEnabled ? "on, present wait"
                          : (!_presentWaitOn ? "off: the device has no present id / present wait, the spare images add "
                                               "queue depth while not generating"
                                             : "off: switched off in the ini or no usable bound"));

    if (_gateEnabled && (_presentMode == VK_PRESENT_MODE_MAILBOX_KHR || _presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR))
        LOG_INFO("F5Low (Vulkan): this game presents uncapped (mode {}), the bound limits it to {} frames ahead",
                 _presentMode, _gateBound);

    // The pacer was made for a latency gate state (it spaces a V-Sync batch by it): a swapchain with the other one gets
    // a new pacer (the worker is quiet here; the next frame makes it)
    if (s.pacer != nullptr && s.pacerGate != _gateEnabled.load())
    {
        LOG_INFO("F5Low (Vulkan): the latency gate is now {}, f5low's pacer is made again for it",
                 _gateEnabled.load() ? "on" : "off");
        s.pacerDestroy(s.pacer);
        s.pacer = nullptr;
        _timer = {};
    }

    _meteringOn = DecideMetering();
    // The swapchain was made with spare images for the most this session generates: a recreate for a raise has nothing
    // more to ask for, so the escalation stays dormant
    _createdWanted = gateWanted ? _maxInterpolationCount : _wanted;
    {
        std::lock_guard askLock(_askMutex);
        _ask = {};
    }
    s.nextSet = 0;
    s.setSlot[0] = s.setSlot[1] = -1;

    // A removed device stays removed; the other latches are re-armed with the new swapchain
    _failures = _removed ? 3 : 0;

    {
        OwnedLockGuard lock(Mutex, 3);
        _pending = {};
    }

    UpdateCount();

    if (_removed)
        SetReason("The D3D12 device was removed");
    else if (_headroom == 0)
        SetReason("The swapchain has no spare images to generate into");
    else
        SetReason("");

    LOG_INFO("F5Low (Vulkan): swapchain {}x{}, format {}, {} images ({} spare), HDR10: {}", _swapchainExtent.width,
             _swapchainExtent.height, magic_enum::enum_name(_swapchainFormat), s.images.size(), _headroom, _hdr10);

    return true;
}

bool F5Low_Vk::DestroySwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
    // The handle is the driver's own, the hook destroys it after this
    if (swapchain == VK_NULL_HANDLE || swapchain != _swapchain)
        return false;

    Deactivate();

    // Our presents wait on our semaphores and show its images
    auto& s = *_impl;
    const bool quiet = QuiesceWorker(2000);

    if (quiet)
    {
        if (device == _device && !s.deviceGone)
        {
            WaitOwnWork();

            if (s.outQueue != VK_NULL_HANDLE && s.outQueue != _gameQueue)
                vkQueueWaitIdle(s.outQueue);
        }

        // A worker that did not stop still reads them
        s.images.clear();
    }

    _swapchain = VK_NULL_HANDLE;
    _headroom = 0;
    _generationOff = true;
    return false;
}

VkResult F5Low_Vk::GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount,
                                      VkImage* pSwapchainImages)
{
    return SwapchainFunction<PFN_vkGetSwapchainImagesKHR>("vkGetSwapchainImagesKHR", vkGetSwapchainImagesKHR)(
        device, swapchain, pSwapchainImageCount, pSwapchainImages);
}

void F5Low_Vk::SetHdrMetadata(VkDevice device, VkSwapchainKHR swapchain, const VkHdrMetadataEXT* pMetadata)
{
    // The hook set _inFGCall around this, so the call goes straight to the driver
    const auto setHdr =
        reinterpret_cast<PFN_vkSetHdrMetadataEXT>(VulkanHooks::GetDeviceProcAddr(device, "vkSetHdrMetadataEXT"));

    if (setHdr != nullptr)
        setHdr(device, 1, &swapchain, pMetadata);
}

// ---- per frame ----

bool F5Low_Vk::Present()
{
    OwnedLockGuard lock(Mutex, 2);

    _pending = {};

    // Game stopped sending new frames, menus and such
    if ((_fgFramePresentId - _lastFGFramePresentId) > 3 && IsActive() && !_waitingNewFrameData)
    {
        LOG_DEBUG("Pausing FG");
        Deactivate();
        _waitingNewFrameData = true;
        return false;
    }

    _fgFramePresentId++;

    UpdateCount(true);

    UINT64 willDispatchFrame = 0;
    const auto index = GetDispatchIndex(willDispatchFrame);

    if (index < 0 || !IsActive() || IsPaused() || !_blocked.empty())
        return false;

    auto velocity = GetResource(FG_ResourceType::Velocity, index);
    auto depth = GetResource(FG_ResourceType::Depth, index);

    if (!velocity || !depth)
    {
        LOG_WARN("Depth or Velocity is not ready, skipping");
        return false;
    }

    _pending.valid = true;
    _pending.index = index;
    _pending.frame = willDispatchFrame;
    _pending.velocity = velocity;
    _pending.depth = depth;
    _pending.hudless = GetResource(FG_ResourceType::HudlessColor, index);
    return true;
}

VkResult F5Low_Vk::QueuePresent(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    // The time between two game presents. The pacer gets the real arrival time (as on D3D12); our own interval is the
    // game's own frame time, the time between two presents without what our waits added to it.
    const double entryMs = Util::MillisecondsNow();
    _arriveTicks = _impl->pacerNowTicks != nullptr ? _impl->pacerNowTicks() : 0;
    const double ourWaitMs = _ourWaitMs.exchange(0.0);

    if (_lastEntryMs > 0.0)
    {
        const double own = Unthrottled(entryMs - _lastEntryMs, ourWaitMs);
        _intervalMs = NextInterval(_intervalMs, own);
        _statWaitMs += ourWaitMs;
    }

    // The stats window runs from the first present after the last line to the last one: the images we presented and the
    // overlay's count are read at the same two moments, so the three rates cover the same time
    if (_statEntries == 0)
    {
        _statFirstEntryMs = entryMs;
        _statShownFirst = _statShown.load();
        _statMeterFirst = MenuCommon::FGOutputPresentTotal(nullptr);
    }

    _statLastEntryMs = entryMs;
    _statShownLast = _statShown.load();
    _statMeterLast = MenuCommon::FGOutputPresentTotal(&_statMeterOn);
    ++_statEntries;

    _lastEntryMs = entryMs;

    Pending pending;

    {
        OwnedLockGuard lock(Mutex, 3);
        pending = std::move(_pending);
        _pending = {};
    }

    // The id this present is known by for the latency bound: the game's own present id, else ours (the present wait
    // needs one on every real present)
    _frameOwnId = 0;
    _inlineBatch = 0;
    _inlineGenerated = 0;

    if (_gateEnabled && pPresentInfo->swapchainCount == 1 && pPresentInfo->pSwapchains[0] == _swapchain)
    {
        uint64_t id = 0;

        for (auto* node = static_cast<const VkBaseInStructure*>(pPresentInfo->pNext); node != nullptr;
             node = node->pNext)
        {
            if (node->sType == VK_STRUCTURE_TYPE_PRESENT_ID_KHR)
            {
                const auto* ids = reinterpret_cast<const VkPresentIdKHR*>(node);

                if (ids->swapchainCount >= 1 && ids->pPresentIds != nullptr)
                    id = ids->pPresentIds[0];
            }
        }

        std::lock_guard gateLock(_gateMutex);
        GateOnPresent(_gate, entryMs);

        // A game id of 0 is no id; ours stay above the game's
        if (id != 0)
            _ownPresentId = OwnIdAfterGame(_ownPresentId, id);
        else
            id = _frameOwnId = ++_ownPresentId;

        NoteIssued(id);
    }

    // No spare image: nothing to copy, nothing to ask of f5low
    if (!pending.valid || !_blocked.empty() || pPresentInfo->swapchainCount != 1 ||
        pPresentInfo->pSwapchains[0] != _swapchain || _failures >= 3 || _headroom == 0)
    {
        _generationOff = true;
        const auto direct = DirectPresent(queue, pPresentInfo, _frameOwnId);
        xefg::AddShownPresent(_shownTotal, 1);
        return AskRecreateIfRaised(direct);
    }

    if (queue != _gameQueue)
    {
        if (!_impl->warnedQueue)
        {
            _impl->warnedQueue = true;
            SetReason("The game presents from a different queue than its first graphics queue");
        }

        _generationOff = true;
        const auto direct = DirectPresent(queue, pPresentInfo, _frameOwnId);
        xefg::AddShownPresent(_shownTotal, 1);
        return AskRecreateIfRaised(direct);
    }

    VkPresentInfoKHR finalInfo = *pPresentInfo;

    // Generate counts its own blocking waits (slot, fences, inline presents), not its CPU work
    const auto outcome = Generate(queue, pPresentInfo, pending, finalInfo);

    if (_generationOff)
        ++_statOffFrames;

    if (outcome == Outcome::Queued)
        return AskRecreateIfRaised(static_cast<VkResult>(_impl->lastResult.load()));

    const auto direct = DirectPresent(queue, &finalInfo, _frameOwnId, _inlineBatch);
    xefg::AddShownPresent(_shownTotal, 1 + _inlineGenerated);
    return AskRecreateIfRaised(direct);
}

// The multiplier was raised past the spare images of this swapchain: the game is told its swapchain is suboptimal once,
// so it makes a new one (CreateSwapchain then asks the driver for the new count). Only on top of a present that went
// through; the result is not kept anywhere else, so it is never taken for the driver's.
VkResult F5Low_Vk::AskRecreateIfRaised(VkResult result)
{
    // Per game present: the 1 ms timer follows generating, and the stats line comes every kStatsEvery frames
    if (auto& s = *_impl; s.pacer != nullptr)
    {
        if (TimerUpdate(_timer, !_generationOff))
            s.pacerTimerRaise(s.pacer, _timer.raised ? 1u : 0u);
    }

    if (++_statFrames >= kStatsEvery)
    {
        _statFrames = 0;

        if (!_generationOff)
            LogStats();
        else
        {
            // Not generating at the boundary: no line, and the window starts over (no half old, half new numbers)
            _statEntries = 0;
            _statOffFrames = 0;
            _statWaitMs = 0.0;

            {
                std::lock_guard lock(_statMutex);
                _spacing = {};
                _statGenerated = 0;
                _statBatches = 0;
            }

            std::lock_guard lock(_gateMutex);
            _gateWaits = 0;
            _gateWaitTotalMs = 0.0;
        }
    }

    if (result != VK_SUCCESS)
        return result;

    std::lock_guard askLock(_askMutex);
    const auto before = _ask.state;

    if (!OnPresentRecreate(_ask, _wanted, _createdWanted, _headroom, Util::MillisecondsNow()))
    {
        if (before != RecreateState::GaveUp && _ask.state == RecreateState::GaveUp)
            LOG_WARN("F5Low (Vulkan): the game did not make a new swapchain for {}x, staying at {} spare images",
                     _wanted + 1, _headroom);

        return result;
    }

    LOG_INFO("F5Low (Vulkan): multiplier {}x needs more spare swapchain images ({} now), asking the game to make a new "
             "swapchain",
             _wanted + 1, _headroom);
    return VK_SUBOPTIMAL_KHR;
}

void F5Low_Vk::WaitOwnWork()
{
    auto& s = *_impl;

    DrainWorker(2000);

    if (_device != VK_NULL_HANDLE && !s.deviceGone)
    {
        for (auto& slot : s.slots)
        {
            VkFence waits[kBatches];
            uint32_t count = 0;

            for (uint32_t i = 0; i < kBatches; i++)
            {
                if (slot.used[i] && slot.fences[i] != VK_NULL_HANDLE)
                    waits[count++] = slot.fences[i];
            }

            if (count > 0)
                vkWaitForFences(_device, count, waits, VK_TRUE, kWaitFenceNs);
        }
    }

    WaitFence12(s.consumed.Fence12(), s.consumed.Value(), 5000);
    WaitFence12(s.done.Fence12(), s.done.Value(), 5000);
    WaitFence12(s.ready.Fence12(), s.ready.Value(), 5000);
    WaitFence12(s.depthDone.Get(), s.depthCounter.issued, 5000);
}

bool F5Low_Vk::EnsureSlots()
{
    auto& s = *_impl;

    if (s.slotsReady)
        return true;

    // The present queue: the extra one PrepareDevice reserved, else the game's (the presents then run on the game
    // thread)
    VkQueue extra = _extraQueues.empty() ? VK_NULL_HANDLE : _extraQueues[0].queue;
    // Swapchain images owned by another queue family would need an ownership transfer: the game's family or none
    const bool sameFamily = extra != VK_NULL_HANDLE && _extraQueues[0].family == _gameQueueFamily;

    if (extra != VK_NULL_HANDLE && !sameFamily)
        LOG_INFO("F5Low (Vulkan): the reserved queue is in family {}, not the game's {}: presents on the game thread",
                 _extraQueues[0].family, _gameQueueFamily);

    s.ownQueue = sameFamily;
    s.outQueue = s.ownQueue ? extra : _gameQueue;
    s.outFamily = _gameQueueFamily;

    if (s.ownQueue)
        LOG_INFO("F5Low (Vulkan): presents on a worker thread, queue family {} index {}", _extraQueues[0].family,
                 _extraQueues[0].index);
    else
        LOG_WARN("F5Low (Vulkan): no extra queue could be reserved, the presents and their pacing run on the game "
                 "thread");

    VkCommandPoolCreateInfo poolInfo { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = _gameQueueFamily;
    bool ok = vkCreateCommandPool(_device, &poolInfo, nullptr, &s.poolIn) == VK_SUCCESS;
    poolInfo.queueFamilyIndex = s.outFamily;
    ok = ok && vkCreateCommandPool(_device, &poolInfo, nullptr, &s.poolOut) == VK_SUCCESS;

    VkSemaphoreCreateInfo semaphoreInfo { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fenceInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

    for (auto& slot : s.slots)
    {
        VkCommandBufferAllocateInfo allocate { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        allocate.commandPool = s.poolIn;
        ok = ok && vkAllocateCommandBuffers(_device, &allocate, &slot.commands[0]) == VK_SUCCESS;
        allocate.commandPool = s.poolOut;
        allocate.commandBufferCount = kMaxGenerated;
        ok = ok && vkAllocateCommandBuffers(_device, &allocate, &slot.commands[1]) == VK_SUCCESS;

        for (auto& fence : slot.fences)
            ok = ok && vkCreateFence(_device, &fenceInfo, nullptr, &fence) == VK_SUCCESS;

        ok = ok && vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &slot.inDone) == VK_SUCCESS;

        for (uint32_t k = 0; k < kMaxGenerated; k++)
        {
            ok = ok && vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &slot.acquired[k]) == VK_SUCCESS;
            ok = ok && vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &slot.outDone[k]) == VK_SUCCESS;
        }
    }

    // Whatever was made is released with the rest (ReleaseAll)
    s.slotsReady = true;

    if (!ok)
    {
        SetReason("Could not make the command buffers or semaphores for the frames");
        return false;
    }

    StartWorker();
    return true;
}

bool F5Low_Vk::Prepare(const Pending& p, uint32_t count, bool& useHudless, uint32_t set)
{
    auto& s = *_impl;
    useHudless = false;

    DXGI_FORMAT dxgiColor = DXGI_FORMAT_UNKNOWN;
    VkFormat vkColor = VK_FORMAT_UNDEFINED;

    if (!native::VkSharedPictureFormat(_swapchainFormat, &dxgiColor, &vkColor))
    {
        SetReason("The swapchain's format cannot be shared with the generator");
        return false;
    }

    const uint32_t width = _swapchainExtent.width;
    const uint32_t height = _swapchainExtent.height;
    std::string why;

    // The D3D12 device the frames are made on: the Vulkan GPU's, the frame source's private one
    if (!s.d3dReady)
    {
        ID3D12Device* device = NativeMotionVk::ShareBridgeDevice(_physicalDevice, why);

        if (device == nullptr)
        {
            SetReason("No D3D12 device on the game's GPU: " + why);
            return false;
        }

        s.device12.Attach(device);

        D3D12_COMMAND_QUEUE_DESC queueDesc {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        bool ok = s.interop.Load(_physicalDevice, _device, why) &&
                  SUCCEEDED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&s.queue12)));

        for (auto& allocator : s.allocators)
            ok = ok &&
                 SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));

        ok = ok && SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.allocators[0].Get(),
                                                       nullptr, IID_PPV_ARGS(&s.list)));

        if (ok)
            s.list->Close();

        ok = ok && s.ready.Create(device, s.interop) && s.consumed.Create(device, s.interop) &&
             s.done.Create(device, s.interop) &&
             SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.depthDone)));
        s.depthCounter = {};

        if (!ok)
        {
            if (why.empty())
                why = !s.ready.Error().empty() ? s.ready.Error() : "creating the D3D12 queue, list or fences failed";

            SetReason("Sharing with the D3D12 device failed: " + why);
            s.ready.Reset();
            s.consumed.Reset();
            s.done.Reset();
            s.depthDone.Reset();
            s.list.Reset();
            s.queue12.Reset();
            s.device12.Reset();
            return false;
        }

        s.d3dReady = true;
        LOG_INFO("F5Low (Vulkan): sharing with a D3D12 device on the game's GPU");
    }

    // The picture and its generated frames
    // Depth and motion flags pick f5low's build variant up front; a later change makes one Generate rebuild
    const uint32_t flags = (_hdr10 ? F5LOW_FRAMES_FLAG_HDR10 : 0) |
                           (IsInvertedDepth() ? F5LOW_FRAMES_FLAG_DEPTH_INVERTED : 0) |
                           (IsJitteredMVs() ? F5LOW_FRAMES_FLAG_MOTION_JITTERED : 0);

    if (!s.color.Matches(width, height, vkColor))
    {
        WaitOwnWork();
        s.hudless.Reset();

        for (auto& outSet : s.out)
        {
            for (auto& image : outSet)
                image.Reset();
        }

        if (!s.color.Create(s.device12.Get(), s.interop, width, height, dxgiColor, vkColor, false))
        {
            SetReason("Sharing the picture failed: " + s.color.Error());
            return false;
        }
    }

    if (s.context != nullptr && (s.contextWidth != width || s.contextHeight != height || s.contextFormat != dxgiColor ||
                                 ((s.contextFlags ^ flags) & F5LOW_FRAMES_FLAG_HDR10) != 0))
    {
        WaitOwnWork();
        const auto result = s.resize(s.context, width, height, dxgiColor, flags);

        // A failed resize is permanent: the context is made again with the next frame
        if (result != F5LOW_FRAMES_OK)
        {
            SetReason(std::string("f5low could not resize its frames: ") + s.lastError(s.context));
            s.destroy(s.context);
            s.context = nullptr;
            return false;
        }

        s.contextWidth = width;
        s.contextHeight = height;
        s.contextFormat = dxgiColor;
        s.contextFlags = flags;
        LOG_INFO("F5Low (Vulkan): frames resized to {}x{}", width, height);
    }

    if (s.context == nullptr)
    {
        F5LowFramesCreateInfo info {};
        info.apiVersion = F5LOW_FRAMES_API_VERSION;
        info.device = s.device12.Get();
        info.width = width;
        info.height = height;
        info.colorFormat = dxgiColor;
        info.maxGenerated = static_cast<uint32_t>(std::min<int>(_maxInterpolationCount, F5LOW_FRAMES_MAX_GENERATED));
        info.flags = flags;
        info.renderWidth = std::min(p.velocity->width, width);
        info.renderHeight = std::min(p.velocity->height, height);

        const auto result = s.create(&info, &s.context);

        if (result != F5LOW_FRAMES_OK || s.context == nullptr)
        {
            SetReason(std::string("f5low could not make its frames: ") + s.lastError(nullptr));
            s.context = nullptr;
            return false;
        }

        s.contextWidth = width;
        s.contextHeight = height;
        s.contextFormat = dxgiColor;
        s.contextFlags = flags;
        LOG_INFO("F5Low (Vulkan): f5low frames made, {}x{}, format {}, up to {} generated", width, height,
                 magic_enum::enum_name(dxgiColor), info.maxGenerated);
    }

    EnsurePacer();

    for (uint32_t k = 0; k < count; k++)
    {
        if (s.out[set][k].Matches(width, height, vkColor))
            continue;

        if (!s.out[set][k].Create(s.device12.Get(), s.interop, width, height, dxgiColor, vkColor, true))
        {
            SetReason("Sharing a generated frame failed: " + s.out[set][k].Error());
            return false;
        }
    }

    // The game's HUD-less picture, when it is the swapchain's size and format
    if (const auto& hudless = p.hudless; hudless && hudless->image != VK_NULL_HANDLE && UsableLayout(hudless->layout) &&
                                         hudless->imageWidth == width && hudless->imageHeight == height)
    {
        DXGI_FORMAT hudlessDxgi = DXGI_FORMAT_UNKNOWN;
        VkFormat hudlessShared = VK_FORMAT_UNDEFINED;

        if (native::VkSharedPictureFormat(hudless->format, &hudlessDxgi, &hudlessShared) && hudlessShared == vkColor)
        {
            if (!s.hudless.Matches(width, height, vkColor))
            {
                WaitOwnWork();

                if (!s.hudless.Create(s.device12.Get(), s.interop, width, height, dxgiColor, vkColor, false))
                {
                    SetReason("Sharing the HUD-less picture failed: " + s.hudless.Error());
                    return false;
                }
            }

            useHudless = true;
        }
    }

    // Which picture the generator gets: the game's hudless one when the input has it and it fits, else colour only
    {
        const auto& hudlessInput = p.hudless;
        const bool offered = hudlessInput && hudlessInput->image != VK_NULL_HANDLE;
        DXGI_FORMAT hudlessDxgi = DXGI_FORMAT_UNKNOWN;
        VkFormat hudlessShared = VK_FORMAT_UNDEFINED;
        const bool formatOk = offered &&
                              native::VkSharedPictureFormat(hudlessInput->format, &hudlessDxgi, &hudlessShared) &&
                              hudlessShared == vkColor;
        const auto mode = HudlessModeOf(
            offered, offered && UsableLayout(hudlessInput->layout),
            offered && hudlessInput->imageWidth == width && hudlessInput->imageHeight == height, formatOk);

        if (static_cast<int>(mode) != _hudlessMode && _hudlessLogs < 8)
        {
            ++_hudlessLogs;
            _hudlessMode = static_cast<int>(mode);
            LOG_INFO("F5Low (Vulkan): {}", HudlessText(mode));
        }
    }

    // Motion vectors cross as they are
    DXGI_FORMAT motionDxgi = DXGI_FORMAT_UNKNOWN;
    const auto& velocity = *p.velocity;

    if (!MotionFormat(velocity.format, &motionDxgi))
    {
        SetReason(std::format("The motion vectors' format {} is not supported (R16G16 and R32G32 float are)",
                              static_cast<int>(velocity.format)));
        return false;
    }

    if (!UsableLayout(velocity.layout) || velocity.image == VK_NULL_HANDLE)
    {
        SetReason("The motion vectors are not in a readable state");
        return false;
    }

    if (!s.motion.Matches(velocity.imageWidth, velocity.imageHeight, velocity.format))
    {
        WaitOwnWork();

        if (!s.motion.Create(s.device12.Get(), s.interop, velocity.imageWidth, velocity.imageHeight, motionDxgi,
                             velocity.format, false))
        {
            SetReason("Sharing the motion vectors failed: " + s.motion.Error());
            return false;
        }
    }

    // Depth: a colour format crosses as an image, a depth format through a buffer and a copy on the D3D12 side
    const auto& depth = *p.depth;

    if (!UsableLayout(depth.layout) || depth.image == VK_NULL_HANDLE)
    {
        SetReason("The depth is not in a readable state");
        return false;
    }

    if (depth.format == VK_FORMAT_R32_SFLOAT)
    {
        s.depthViaBuffer = false;

        if (!s.depthImage.Matches(depth.imageWidth, depth.imageHeight, depth.format))
        {
            WaitOwnWork();

            if (!s.depthImage.Create(s.device12.Get(), s.interop, depth.imageWidth, depth.imageHeight,
                                     DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, false))
            {
                SetReason("Sharing the depth failed: " + s.depthImage.Error());
                return false;
            }
        }
    }
    else
    {
        native::VkDepthCopyFormat format {};

        if (!native::VkDepthCopyFormatOf(depth.format, &format))
        {
            SetReason(std::format("The depth format {} is not supported", static_cast<int>(depth.format)));
            return false;
        }

        s.depthViaBuffer = true;

        if (s.depthSource != depth.format || s.depthWidth != depth.imageWidth || s.depthHeight != depth.imageHeight ||
            s.depthTexture == nullptr)
        {
            WaitOwnWork();
            s.depthTexture.Reset();

            const uint64_t size =
                static_cast<uint64_t>(native::DepthRowPitch(depth.imageWidth, format.bytes)) * depth.imageHeight;

            if (!s.depthBuffer.Create(s.device12.Get(), s.interop, size))
            {
                SetReason("Sharing the depth failed: " + s.depthBuffer.Error());
                return false;
            }

            // Read by f5low's own queue in the common state: simultaneous access lets it be promoted there
            D3D12_HEAP_PROPERTIES heap {};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc {};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = depth.imageWidth;
            desc.Height = depth.imageHeight;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = format.texture;
            desc.SampleDesc.Count = 1;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

            if (FAILED(s.device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&s.depthTexture))))
            {
                SetReason("Could not make the depth texture on the D3D12 device");
                return false;
            }

            s.depthFormat = format;
            s.depthSource = depth.format;
            s.depthWidth = depth.imageWidth;
            s.depthHeight = depth.imageHeight;
            LOG_INFO("F5Low (Vulkan): sharing the depth at {}x{}, Vulkan format {} ({})", s.depthWidth, s.depthHeight,
                     static_cast<int>(depth.format), FormatName(depth.format));
        }
    }

    return true;
}

// The generated frames of one job, spaced by the game's frame interval: for each, acquire an image, copy the shared
// output into it, present. The pacing waits are in here and nowhere that holds the swapchain lock.
// Whether the game's next frame has its copies done already (a job queued behind this one whose copy in finished): what
// f5lowPacerStale reads as "the next game frame is finished on the GPU"
bool F5Low_Vk::NextGameFrameDone()
{
    auto& s = *_impl;
    VkFence fence = VK_NULL_HANDLE;

    {
        std::lock_guard lock(s.jobMutex);

        if (!s.jobs.empty())
            fence = s.slots[s.jobs.front().slot].fences[0];
    }

    return fence != VK_NULL_HANDLE && vkGetFenceStatus(_device, fence) == VK_SUCCESS;
}

// Before frame k (1-based; generated + 1 is the game's own): where on the schedule it goes and the wait for it. True:
// the generated frame is stale and left out (told to the pacer as dropped by the caller).
bool F5Low_Vk::PacerStart(PacerCtx& pc, uint32_t k, bool generatedFrame)
{
    auto& s = *_impl;
    F5LowPacerTarget target {};
    target.structSize = sizeof(target);

    if (s.pacerTarget(s.pacer, &pc.batch, k, &target) != F5LOW_PACER_OK)
        return false;

    if (generatedFrame && s.pacerStale(s.pacer, &pc.batch, k, s.pacerNowTicks(), NextGameFrameDone() ? 1u : 0u) == 1)
    {
        LOG_DEBUG("F5Low (Vulkan): generated frame {} is stale, left out", k);
        return true;
    }

    // Sleep to its release time, then (the present waits for the GPU through its semaphore) present
    // (never past two intervals or 100 ms from now: a wrong schedule must not hold the worker)
    if ((target.flags & F5LOW_PACER_TARGET_PACED) != 0 && target.releaseTicks != 0)
    {
        // The clamp is two intervals of the pacer's own (from the batch it planned); our _intervalMs is the game's CPU
        // time only
        const double pacerIntervalMs = PacerIntervalMs(pc.batch.stepTicks, pc.batch.generated, s.ticksPerSecond);
        s.pacerWaitUntil(ClampRelease(target.releaseTicks, s.pacerNowTicks(), s.ticksPerSecond,
                                      pacerIntervalMs > 0.0 ? pacerIntervalMs : _intervalMs.load()));
    }

    return false;
}

void F5Low_Vk::PacerPresented(PacerCtx& pc, const Job& job, uint32_t k, bool dropped, VkResult result,
                              uint32_t feedback)
{
    auto& s = *_impl;
    const int64_t now = s.pacerNowTicks();

    // The driver's answer to the batch's metering config comes with its first present
    if (job.metered && !pc.meterReported && !dropped)
    {
        pc.meterReported = true;
        s.pacerMeterResult(s.pacer, &pc.batch,
                           MeterStatus(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR, feedback), 0);
    }

    F5LowPacerPresented what {};
    what.structSize = sizeof(what);
    what.presentTicks = now;
    // f5low's work for the frame is done by the time a generated frame is shown, but the watcher may not have noted it
    // yet: a batch with a lead waits a moment for it (the lateness of exactly the late frames is what lengthens the
    // lead); the others take what is there
    what.gpuDoneTicks =
        WatchTake(s, StampBook::Done, job.slot, job.doneValue, DoneStampWaitMs(pc.batch.leadTicks, k, job.generated));
    what.flags = dropped ? F5LOW_PACER_PRESENT_DROPPED : 0;
    s.pacerPresented(s.pacer, &pc.batch, k, &what);

    if (!dropped && s.ticksPerSecond > 0)
    {
        std::lock_guard lock(_statMutex);

        if (_lastPresentTicks != 0)
        {
            const double ms =
                static_cast<double>(now - _lastPresentTicks) * 1000.0 / static_cast<double>(s.ticksPerSecond);
            _spacing.Add(ms);

            const double step =
                static_cast<double>(pc.batch.stepTicks) * 1000.0 / static_cast<double>(s.ticksPerSecond);
            const double slot =
                static_cast<double>(now - pc.batch.baseTicks) * 1000.0 / static_cast<double>(s.ticksPerSecond);

            if (Late(slot, step * k, step))
                ++_spacing.late;
        }

        _lastPresentTicks = now;
    }
}

// f5low's pacer is made once the generator's context is (its GPU gives the pacer the metering rules); the host says it
// can meter only with VK_NV_present_metering on the device, and never that it can burst
void F5Low_Vk::EnsurePacer()
{
    auto& s = *_impl;

    if (!s.pacerExports || s.pacer != nullptr || s.context == nullptr)
        return;

    if (!Config::Instance()->F5LowVulkanPacer.value_or_default())
    {
        static bool logged = false;

        if (!logged)
        {
            logged = true;
            LOG_INFO("F5Low (Vulkan): [F5Low] VulkanPacer=false, plain spacing and the fallback metering rule");
        }

        return;
    }

    _hostMeter = HostCanMeter(Config::Instance()->F5LowVulkanFlipMetering.value_or_default(), _meteringAvailable);

    F5LowPacerCreateInfo info {};
    info.apiVersion = F5LOW_PACER_API_VERSION;
    info.structSize = sizeof(info);
    info.flags = PacerFlags(_hostMeter);
    info.ticksPerSecond = 0;
    info.gpuSm = s.gpuSm != nullptr ? s.gpuSm(s.context) : 0;
    info.maxGenerated =
        static_cast<uint32_t>(std::min(_maxInterpolationCount, static_cast<int>(F5LOW_PACER_MAX_GENERATED)));
    info.refreshHz = DisplayRefreshFps(Util::MillisecondsNow());

    // Who spaces a V-Sync / FIFO batch, as on D3D12 (pace_sched.h FixedTimer): with the latency gate on the display
    // does (the batch goes to the driver back to back, the gate paces by itself), without it f5low's FixedPacing
    // setting decides (auto = timer). MAILBOX / IMMEDIATE batches are timed either way. The fields exist from API level
    // 2; an older f5low.dll ignores them and times every batch.
    const bool gate = _gateEnabled.load();
    info.fixedPacing = 0;
    info.gate = gate ? 1u : 0u;

    if (s.pacerCreate(&info, &s.pacer) != F5LOW_PACER_OK || s.pacer == nullptr)
    {
        s.pacer = nullptr;
        LOG_WARN("F5Low (Vulkan): f5low's pacer could not be made, plain spacing");
        return;
    }

    s.ticksPerSecond = s.pacerTicksPerSecond();
    s.pacerGate = gate;
    const auto level = reinterpret_cast<uint32_t (*)()>(GetProcAddress(s.module, "f5lowPacerApiLevel"));
    const uint32_t apiLevel = level != nullptr ? level() : 1u;
    LOG_INFO(
        "F5Low (Vulkan): f5low's pacer made (sm_{}, flags 0x{:X}), the batches' times and the metering come from it; "
        "V-Sync / FIFO batches: {}",
        info.gpuSm, info.flags,
        apiLevel < 2 ? "timer (this f5low.dll is older than the FixedPacing / gate fields: it times every batch)"
        : DisplaySpaces(gate, VK_PRESENT_MODE_FIFO_KHR, 0)
            ? "display (the latency gate is on: back to back, the display spaces them)"
            : "timer (no latency gate; f5low.ini FixedPacing=display would change it)");
}

// One line every kStatsEvery game frames: what the output did
void F5Low_Vk::LogStats()
{
    auto& s = *_impl;
    F5LowPacerInfo info {};
    info.structSize = sizeof(info);
    const bool haveInfo = s.pacer != nullptr && s.pacerGetInfo(s.pacer, &info) == F5LOW_PACER_OK;

    SpacingStats spacing;
    uint64_t generated = 0, batches = 0;

    {
        std::lock_guard lock(_statMutex);
        spacing = _spacing;
        generated = _statGenerated;
        batches = _statBatches;
        _spacing = {};
        _statGenerated = 0;
        _statBatches = 0;
    }

    uint64_t gateWaits = 0;
    double gateMs = 0.0;

    {
        std::lock_guard lock(_gateMutex);
        gateWaits = _gateWaits;
        gateMs = _gateWaitTotalMs;
        _gateWaits = 0;
        _gateWaitTotalMs = 0.0;
    }

    // Everything below is for this window: the game presents since the last line
    const double spanMs = _statLastEntryMs - _statFirstEntryMs;
    const uint32_t entries = _statEntries;
    const uint32_t offFrames = _statOffFrames;
    const double waitMs = _statWaitMs;
    const uint64_t shown = _statShownLast - _statShownFirst;
    const uint64_t meterShown = _statMeterLast - _statMeterFirst;
    _statEntries = 0;
    _statOffFrames = 0;
    _statWaitMs = 0.0;

    // The real game rate from the real arrival times (entries - 1 intervals); every image we presented in the window,
    // over the same span (the images presented while the window's last frame was still being shown count in the next
    // window, so the span runs from the first to the last game present)
    const bool measured = entries >= 2 && spanMs > 0.0;
    const double gameFps = measured ? (entries - 1) * 1000.0 / spanMs : 0.0;
    const double shownFps = measured ? static_cast<double>(shown) * 1000.0 / spanMs : 0.0;
    const double overlayFps = measured ? static_cast<double>(meterShown) * 1000.0 / spanMs : 0.0;
    const double otherWaitMs = std::max(0.0, waitMs - gateMs);

    LOG_INFO(
        "F5Low (Vulkan) stats ({} game frames, {} not generated): game {:.1f} fps, shown {:.1f} fps, generated "
        "{:.2f}/frame, our waits {:.2f} ms/frame (gate {:.2f}, other {:.2f}), interval {:.2f} ms, pacing {} (mode {}, "
        "active {}), metering {} (feedback 0x{:X}, refused {}), present spacing mean {:.2f} sd {:.2f} max {:.2f} ms, "
        "late {}, gate waits {}; overlay meter {} ({:.1f} fps counted, FG active {}, paused {}); present mode {}{}",
        entries, offFrames, gameFps, shownFps,
        batches > 0 ? static_cast<double>(generated) / static_cast<double>(batches) : 0.0,
        entries > 0 ? waitMs / entries : 0.0, entries > 0 ? gateMs / entries : 0.0,
        entries > 0 ? otherWaitMs / entries : 0.0, _intervalMs.load(), s.pacer != nullptr ? "f5low pacer" : "plain",
        haveInfo ? info.pacingMode : -1, haveInfo ? info.pacingActive : 0,
        _meteringOn.load() || _meterWasOn.load() ? "on" : "off", _lastFeedback.load(),
        haveInfo ? info.meteringRefused : 0, spacing.mean, spacing.Sd(), spacing.max, spacing.late, gateWaits,
        _statMeterOn ? "on" : "off", overlayFps, IsActive() ? "yes" : "no", IsPaused() ? "yes" : "no", _presentMode,
        _tearing ? std::format(" (game asked {}: tearing like D3D12)", _gamePresentMode) : std::string());
}

uint32_t F5Low_Vk::RunGenerated(const Job& job, bool paced, PacerCtx* pacer)
{
    auto& s = *_impl;
    auto& slot = s.slots[job.slot];
    uint32_t presentedGenerated = 0;
    // No CPU spacing when V-Sync / FIFO paces the display; else the batch fits the game's own interval
    const double spacingMs = BatchSpacingMs(job.intervalMs, job.generated, VsyncPaces(EffectiveMode(job)));
    const double baseMs = Util::MillisecondsNow();
    bool complete = true;

    VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    for (uint32_t k = 0; k < job.generated; k++)
    {
        // f5low's pacer places the frame (and leaves a stale one out); without it the plain spacing
        if (pacer != nullptr)
        {
            if (PacerStart(*pacer, k + 1, true))
            {
                PacerPresented(*pacer, job, k + 1, true, VK_SUCCESS, 0);
                continue;
            }
        }
        else if (paced && k > 0 && spacingMs > 0.0)
        {
            WaitUntil(baseMs + k * spacingMs);
        }

        uint32_t shownIndex = 0;

        if (!AcquireImage(slot.acquired[k], &shownIndex, kAcquireMs) || shownIndex >= s.images.size())
        {
            LOG_WARN("F5Low (Vulkan): no swapchain image for generated frame {}", k);
            complete = false;
            break;
        }

        VkCommandBuffer cmd = slot.commands[1 + k];
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &begin);
        RecordOutput(cmd, s.out[job.set][k], s.images[shownIndex], s.outFamily);
        vkEndCommandBuffer(cmd);

        const VkSemaphore waits[2] = { slot.acquired[k], s.done.Semaphore() };
        const VkPipelineStageFlags stages[2] = { VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
        const uint64_t waitValues[2] = { 0, job.doneValue };
        const uint64_t signalValue = 0;

        VkTimelineSemaphoreSubmitInfo timeline { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timeline.waitSemaphoreValueCount = 2;
        timeline.pWaitSemaphoreValues = waitValues;
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &signalValue;

        VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline };
        submit.waitSemaphoreCount = 2;
        submit.pWaitSemaphores = waits;
        submit.pWaitDstStageMask = stages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &slot.outDone[k];

        const auto submitted = vkQueueSubmit(s.outQueue, 1, &submit, slot.fences[1 + k]);
        slot.used[1 + k] = submitted == VK_SUCCESS;

        // The game's device group present info (one device) goes with every present of the frame
        VkDeviceGroupPresentInfoKHR groupInfo { VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR };
        groupInfo.swapchainCount = job.groupCount;
        groupInfo.pDeviceMasks = &job.groupMask;
        groupInfo.mode = static_cast<VkDeviceGroupPresentModeFlagBitsKHR>(job.groupMode);

        // A metered batch carries its planned size on every present; the first present of a plain batch after metered
        // ones closes the run
        const uint32_t batch = job.metered ? job.generated + 1 : (k == 0 && _meterWasOn.exchange(false) ? 1u : 0u);
        VkSetPresentConfigNV meterConfig { kStructSetPresentConfigNV };
        meterConfig.numFramesPerBatch = batch;
        meterConfig.pNext = job.hasGroup ? &groupInfo : nullptr;

        VkPresentInfoKHR shown { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        shown.pNext = batch != 0 ? static_cast<const void*>(&meterConfig) : (job.hasGroup ? &groupInfo : nullptr);
        shown.waitSemaphoreCount = 1;
        shown.swapchainCount = 1;
        shown.pSwapchains = &_swapchain;
        shown.pImageIndices = &shownIndex;

        if (submitted != VK_SUCCESS)
        {
            // The acquired image goes back through a present (waiting for its acquire semaphore) so it is not lost
            LOG_ERROR("F5Low (Vulkan): vkQueueSubmit for generated frame {} failed: {}", k,
                      magic_enum::enum_name(submitted));
            shown.pWaitSemaphores = &slot.acquired[k];
            PresentOn(s.outQueue, shown, kWorkerLockMs);
            complete = false;
            break;
        }

        shown.pWaitSemaphores = &slot.outDone[k];

        const auto presented = PresentOn(s.outQueue, shown, kWorkerLockMs);

        if (job.metered)
            NoteMeterFeedback(presented, meterConfig.presentConfigFeedback);

        if (presented == VK_SUCCESS || presented == VK_SUBOPTIMAL_KHR)
            ++presentedGenerated;
        else if (job.metered && presented != VK_TIMEOUT)
            NoteMeterFeedback(presented, 0);

        if (pacer != nullptr)
            PacerPresented(*pacer, job, k + 1, presented == VK_TIMEOUT, presented, meterConfig.presentConfigFeedback);

        if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR)
        {
            LOG_WARN("F5Low (Vulkan): presenting generated frame {} returned {}", k, magic_enum::enum_name(presented));
            s.lastResult = presented;
            complete = false;
            break;
        }
    }

    // The game's frame follows after the last generated one's slot
    if (paced && complete && spacingMs > 0.0)
        WaitUntil(baseMs + job.generated * spacingMs);

    return presentedGenerated;
}

F5Low_Vk::Outcome F5Low_Vk::Generate(VkQueue queue, const VkPresentInfoKHR* present, Pending& p,
                                     VkPresentInfoKHR& finalInfo)
{
    auto& s = *_impl;
    const double startMs = Util::MillisecondsNow();

    // What this thread may wait for in one present, in all
    const auto leftMs = [&] { return RemainingMs(kBlockMs, Util::MillisecondsNow() - startMs); };

    // Our blocking waits on this thread, counted into the game's own frame time
    const auto timed = [&](auto&& wait)
    {
        const double waitStartMs = Util::MillisecondsNow();
        const auto result = wait();
        AddOurWait(Util::MillisecondsNow() - waitStartMs);
        return result;
    };

    // Assumed off until a frame really was generated: the cap and the status follow what happens
    _generationOff = true;

    if (!p.depth || !p.velocity || s.images.empty() || present->pImageIndices == nullptr)
        return Outcome::Direct;

    const uint32_t imageIndex = present->pImageIndices[0];

    if (imageIndex >= s.images.size())
        return Outcome::Direct;

    // What the game chained to its present has to be carried by the real present of a generated frame
    Job job;

    if (!CarryPresentStructs(present, job))
        return Outcome::Direct;

    const VkImage source = s.images[imageIndex];

    uint32_t count = CountToGenerate(
        DynamicHold(),
        GeneratedCount(_wanted, static_cast<uint32_t>(std::min(_maxInterpolationCount, _countLimit)), _headroom));

    // Until the frame interval is known the generator only takes the frames as history
    if (!CanPace(_intervalMs, count))
        count = 0;

    // Two sets of output images: the one this frame writes was last used by the job before the previous one
    const uint32_t set = s.nextSet;
    bool useHudless = false;

    if (!Prepare(p, count, useHudless, set) || !EnsureSlots())
    {
        _failures++;
        return Outcome::Direct;
    }

    StartWorker();

    // f5low's fences: the game queue never waits on a value that will not arrive (a removed device, a stuck generator)
    {
        const auto consumedDone = s.consumed.Fence12()->GetCompletedValue();
        const auto doneDone = s.done.Fence12()->GetCompletedValue();

        if (FenceBehind(consumedDone, s.consumed.Value()) || FenceBehind(doneDone, s.done.Value()))
        {
            if (consumedDone == UINT64_MAX || doneDone == UINT64_MAX)
            {
                _removed = true;
                LatchOff("The D3D12 device was removed");
            }
            else if (!_loggedBehind)
            {
                _loggedBehind = true;
                LOG_WARN("F5Low (Vulkan): f5low is more than a frame behind, passing frames through until it "
                         "catches up");
            }

            return Outcome::Direct;
        }
    }

    const uint32_t slotIndex = s.cursor;
    auto& slot = s.slots[slotIndex];

    // The slot's last use is over: its command buffers and semaphores are ours again
    {
        std::unique_lock lock(s.jobMutex);

        if (!timed(
                [&]
                {
                    return s.jobCv.wait_for(lock, std::chrono::milliseconds(leftMs()),
                                            [&] { return !s.slotBusy[slotIndex]; });
                }))
        {
            LatchOff("The present worker did not finish a frame in time, frame generation is off");
            return Outcome::Direct;
        }
    }

    {
        VkFence waits[kBatches];
        uint32_t waitCount = 0;

        for (uint32_t i = 0; i < kBatches; i++)
        {
            if (slot.used[i])
                waits[waitCount++] = slot.fences[i];
        }

        if (waitCount > 0)
        {
            if (timed(
                    [&]
                    {
                        return vkWaitForFences(_device, waitCount, waits, VK_TRUE,
                                               static_cast<uint64_t>(leftMs()) * 1'000'000ull);
                    }) != VK_SUCCESS)
            {
                LatchOff("The GPU did not finish a frame's copies in time, frame generation is off");
                return Outcome::Direct;
            }

            vkResetFences(_device, waitCount, waits);

            for (auto& used : slot.used)
                used = false;
        }
    }

    s.cursor = (s.cursor + 1) % kSlots;

    const auto& velocity = *p.velocity;
    const auto& depth = *p.depth;

    // The copies in: the presented picture, the HUD-less one, motion and depth, into the shared images
    VkCommandBuffer cmd = slot.commands[0];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    native::RecordCopyToShared(cmd, source, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, s.color, _gameQueueFamily);

    if (useHudless)
        native::RecordCopyToShared(cmd, p.hudless->image, p.hudless->layout, s.hudless, _gameQueueFamily);

    native::RecordCopyToShared(cmd, velocity.image, velocity.layout, s.motion, _gameQueueFamily);

    if (s.depthViaBuffer)
        native::RecordDepthToBuffer(cmd, depth.image, depth.format, depth.layout, depth.imageWidth, depth.imageHeight,
                                    s.depthBuffer.Buffer());
    else
        native::RecordCopyToShared(cmd, depth.image, depth.layout, s.depthImage, _gameQueueFamily);

    vkEndCommandBuffer(cmd);

    // Waits: what the game's present waited on (this submit takes it over), and f5low's last read of the shared images
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<uint64_t> waitValues;

    for (uint32_t i = 0; i < present->waitSemaphoreCount; i++)
    {
        waitSemaphores.push_back(present->pWaitSemaphores[i]);
        waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        waitValues.push_back(0);
    }

    waitSemaphores.push_back(s.consumed.Semaphore());
    waitStages.push_back(VK_PIPELINE_STAGE_TRANSFER_BIT);
    waitValues.push_back(s.consumed.Value());

    // Taken once the submit went through
    const uint64_t copiedValue = s.ready.Value() + 1;
    const VkSemaphore signalSemaphores[2] = { s.ready.Semaphore(), slot.inDone };
    const uint64_t signalValues[2] = { copiedValue, 0 };

    VkTimelineSemaphoreSubmitInfo timeline { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    timeline.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
    timeline.pWaitSemaphoreValues = waitValues.data();
    timeline.signalSemaphoreValueCount = 2;
    timeline.pSignalSemaphoreValues = signalValues;

    VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline };
    submit.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
    submit.pWaitSemaphores = waitSemaphores.data();
    submit.pWaitDstStageMask = waitStages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 2;
    submit.pSignalSemaphores = signalSemaphores;

    if (const auto result = vkQueueSubmit(queue, 1, &submit, slot.fences[0]); result != VK_SUCCESS)
    {
        SetReason(std::string("vkQueueSubmit failed: ") + std::string(magic_enum::enum_name(result)));
        _failures++;
        return Outcome::Direct;
    }

    s.ready.Next();
    slot.used[0] = true;

    // From here the game's present waits on our copy in, whatever else happens
    finalInfo.waitSemaphoreCount = 1;
    finalInfo.pWaitSemaphores = &slot.inDone;

    uint64_t readyValue = copiedValue;
    ID3D12Fence* readyFence = s.ready.Fence12();

    // The depth buffer into its texture on the D3D12 side, after the copy in; its own fence tells f5low it is there
    if (s.depthViaBuffer)
    {
        auto* allocator = s.allocators[slotIndex].Get();

        if (!timed([&] { return WaitFence12(s.depthDone.Get(), s.allocatorValue[slotIndex], leftMs()); }) ||
            FAILED(allocator->Reset()) || FAILED(s.list->Reset(allocator, nullptr)))
        {
            SetReason("The D3D12 depth copy could not be recorded");
            _failures++;
            return Outcome::Direct;
        }

        const auto transition = [&](D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
        {
            D3D12_RESOURCE_BARRIER barrier {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = s.depthTexture.Get();
            barrier.Transition.StateBefore = from;
            barrier.Transition.StateAfter = to;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            s.list->ResourceBarrier(1, &barrier);
        };

        transition(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        native::RecordBufferToDepthTexture(s.list.Get(), s.depthBuffer.Res12(), s.depthTexture.Get(), s.depthFormat,
                                           s.depthWidth, s.depthHeight);
        transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        s.list->Close();

        s.queue12->Wait(s.ready.Fence12(), copiedValue);
        ID3D12CommandList* lists[] = { s.list.Get() };
        s.queue12->ExecuteCommandLists(1, lists);

        readyValue = s.depthCounter.Take();
        s.queue12->Signal(s.depthDone.Get(), readyValue);
        s.allocatorValue[slotIndex] = readyValue;
        readyFence = s.depthDone.Get();
    }

    // When the game's frame is finished on the GPU, for the pacer
    if (count > 0)
        WatchPush(s, StampBook::Ready, readyFence, readyValue, StampKey(readyValue, s.depthViaBuffer), slotIndex);

    // The job two frames back (the last one to write this set of output images) is done with them before f5low writes
    // them again
    if (count > 0 && s.setSlot[set] >= 0)
    {
        const uint32_t previous = static_cast<uint32_t>(s.setSlot[set]);
        {
            std::unique_lock lock(s.jobMutex);

            if (!timed(
                    [&]
                    {
                        return s.jobCv.wait_for(lock, std::chrono::milliseconds(leftMs()),
                                                [&] { return !s.slotBusy[previous]; });
                    }))
            {
                LatchOff("The present worker did not finish a frame in time, frame generation is off");
                return Outcome::Direct;
            }
        }

        VkFence waits[kMaxGenerated];
        uint32_t waitCount = 0;

        for (uint32_t i = 0; i < kMaxGenerated; i++)
        {
            if (s.slots[previous].used[1 + i])
                waits[waitCount++] = s.slots[previous].fences[1 + i];
        }

        if (waitCount > 0 && timed(
                                 [&]
                                 {
                                     return vkWaitForFences(_device, waitCount, waits, VK_TRUE,
                                                            static_cast<uint64_t>(leftMs()) * 1'000'000ull);
                                 }) != VK_SUCCESS)
        {
            LatchOff("The GPU did not finish the last generated frames in time, frame generation is off");
            return Outcome::Direct;
        }
    }

    // f5low's call: the frames between the previous pair and this one
    F5LowFramesInput input {};
    input.color = s.color.Res12();
    input.hudless = useHudless ? s.hudless.Res12() : nullptr;
    input.motion = s.motion.Res12();
    input.motionWidth = std::min({ velocity.width, velocity.imageWidth, _swapchainExtent.width });
    input.motionHeight = std::min({ velocity.height, velocity.imageHeight, _swapchainExtent.height });
    input.depth = s.depthViaBuffer ? s.depthTexture.Get() : s.depthImage.Res12();
    input.depthWidth = std::min({ depth.width, depth.imageWidth, _swapchainExtent.width });
    input.depthHeight = std::min({ depth.height, depth.imageHeight, _swapchainExtent.height });

    const int index = p.index;
    input.motionScale[0] = _mvScaleX[index] != 0.0f ? _mvScaleX[index] : 1.0f;
    input.motionScale[1] = _mvScaleY[index] != 0.0f ? _mvScaleY[index] : 1.0f;
    input.jitter[0] = _jitterX[index];
    input.jitter[1] = _jitterY[index];
    input.cameraNear = _cameraNear[index];
    input.cameraFar = _cameraFar[index];
    input.cameraFovY = _cameraVFov[index];
    input.depthInverted = IsInvertedDepth() ? 1 : 0;
    input.motionJittered = IsJitteredMVs() ? 1 : 0;
    input.reset = _reset[index] != 0 ? 1 : 0;
    input.frameId = p.frame;
    input.readyFence = readyFence;
    input.readyValue = readyValue;
    input.consumedFence = s.consumed.Fence12();
    input.consumedValue = s.consumed.Value() + 1;

    F5LowFramesOutput output {};
    output.count = count;

    for (uint32_t k = 0; k < count; k++)
        output.targets[k] = s.out[set][k].Res12();

    output.doneFence = s.done.Fence12();
    output.doneValue = s.done.Value() + 1;

    // f5low signals the done fence for every answer, so the stamp always comes
    if (count > 0)
        WatchPush(s, StampBook::Done, output.doneFence, output.doneValue, output.doneValue, slotIndex);

    const auto result = s.generate(s.context, &input, &output);

    // f5low signals both fences for every answer (also NOT_READY and errors), in order on its queue
    s.consumed.Next();
    s.done.Next();

    if (result != F5LOW_FRAMES_OK)
    {
        if (result == F5LOW_FRAMES_NOT_READY)
        {
            if (!s.warnedNotReady)
            {
                s.warnedNotReady = true;
                LOG_INFO("F5Low (Vulkan): the generator is still being built or busy, presenting the game's frames");
            }
        }
        else if (result == F5LOW_FRAMES_ERROR_UNSUPPORTED && count > 1)
        {
            _countLimit = 1;
            LOG_WARN("F5Low (Vulkan): this generator only makes one frame per game frame, limiting to 2x");
        }
        else
        {
            _failures++;
            SetReason(std::format("f5low refused a frame ({}): {}", static_cast<int>(result), s.lastError(s.context)));
        }
    }
    else
    {
        _failures = 0;
        SetReason(static_cast<uint32_t>(_wanted) > _headroom
                      ? std::format("Only {} spare swapchain images: generating fewer frames than the multiplier asks "
                                    "for until the game makes a new swapchain",
                                    _headroom)
                      : std::string());
    }

    const uint32_t generated = GeneratedAfter(result, output.generated, count);

    if (generated == 0)
        return Outcome::Direct;

    _generationOff = false;

    if (!_loggedGenerating)
    {
        _loggedGenerating = true;
        LOG_INFO("F5Low (Vulkan): generating {} frame(s) per game frame, frame interval {:.2f} ms", generated,
                 _intervalMs.load());
    }

    job.slot = slotIndex;
    job.set = set;
    job.generated = generated;
    job.doneValue = output.doneValue;
    job.readyValue = StampKey(readyValue, s.depthViaBuffer);
    job.realIndex = imageIndex;
    job.realWait = slot.inDone;
    job.intervalMs = _intervalMs;
    _gateCount = generated;
    _gateMode = EffectiveMode(job);
    job.metered = _meteringOn;
    job.arriveTicks = _arriveTicks;

    // The present wait needs an id on the real present: the game's (carried) or ours
    if (!job.hasPresentId && _frameOwnId != 0)
    {
        job.hasPresentId = true;
        job.presentId = _frameOwnId;
    }

    s.setSlot[set] = static_cast<int>(slotIndex);
    s.nextSet ^= 1;

    if (s.worker.joinable())
    {
        {
            std::lock_guard lock(s.jobMutex);
            s.jobs.push_back(job);
            s.slotBusy[slotIndex] = true;
        }

        s.jobCv.notify_all();
        return Outcome::Queued;
    }

    // No present queue of our own: the presents and their pacing run here, the real image goes out after (in the same
    // metered batch)
    const double inlineStartMs = Util::MillisecondsNow();
    const uint32_t presentedInline = RunGenerated(job, PacedOnCpu(job.metered));
    AddOurWait(Util::MillisecondsNow() - inlineStartMs); // the game thread is held for all of it
    _inlineGenerated = presentedInline;
    _inlineBatch = job.metered ? presentedInline + 1 : 0;
    _meterWasOn = job.metered;
    return Outcome::Direct;
}

// Where f5low's pacer export (Story 3) may answer the metering question; until then (and with an older f5low.dll) it is
// off unless FlipMetering=1 forces it on (XEFG_CUDA_FLIPMETERING or f5low.ini, as f5low reads it), never on a V-Sync
// swapchain (FIFO paces the presents, as DisplayPaced on D3D12) and never once the driver's feedback turned it off.
bool F5Low_Vk::DecideMetering()
{
    std::string setting = "auto";

    if (char value[64] {}; GetEnvironmentVariableA("XEFG_CUDA_FLIPMETERING", value, sizeof(value)) != 0)
    {
        setting = value;
    }
    else
    {
        std::error_code ec;
        const auto ini = std::filesystem::absolute(F5Low::LibraryPath(), ec).parent_path() / L"f5low.ini";
        wchar_t wide[64] {};
        GetPrivateProfileStringW(L"f5low", L"FlipMetering", L"auto", wide, 64, ini.c_str());
        setting.clear();

        for (const wchar_t c : wide)
        {
            if (c == 0)
                break;

            setting += static_cast<char>(c);
        }
    }

    const int forced = FlipMeteringForced(setting.c_str());
    const bool vsync = VsyncPaces(_presentMode);
    bool refused = false;

    {
        std::lock_guard gateLock(_gateMutex);
        refused = _meterHealth.off;
    }

    const bool on = !refused && MeteringOn(forced, _meteringAvailable, vsync);

    if (_impl->pacerExports)
        LOG_INFO("F5Low (Vulkan): with f5low's pacer the batches are metered when it asks for it (FlipMetering in "
                 "f5low.ini, "
                 "the GPU, V-Sync); the line below is the fallback rule only");

    LOG_INFO("F5Low (Vulkan): flip metering {} (FlipMetering={}, VK_NV_present_metering {}, present mode {}{}{})",
             on ? "on" : "off", setting, _meteringAvailable ? "enabled" : "not on the device", _presentMode,
             vsync ? ", V-Sync paces the presents" : "", refused ? ", refused by the driver before" : "");

    return on;
}

void F5Low_Vk::NoteIssued(uint64_t id)
{
    // _gateMutex is held by the caller
    _issuedIds[_issuedCount % F5LowVk::kGateRing] = id;
    ++_issuedCount;
}

// The driver's answer to a metered present: logged when it changes, and after kMeterRefusals bad ones in a row metering
// is off for the session (CPU spacing from then on). What a non-zero feedback means is not known yet, so it counts
// against it.
void F5Low_Vk::NoteMeterFeedback(VkResult result, uint32_t feedback)
{
    const bool failed = result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR;
    std::lock_guard gateLock(_gateMutex);

    if ((feedback != _lastFeedback || failed) && _meterLogs < 32)
    {
        ++_meterLogs;
        LOG_INFO("F5Low (Vulkan): flip metering feedback 0x{:X}, present result {}", feedback,
                 magic_enum::enum_name(result));
    }

    _lastFeedback = feedback;
    const bool wasOff = _meterHealth.off;
    MeterNote(_meterHealth, MeterNegative(failed, feedback));

    if (!wasOff && _meterHealth.off)
    {
        _meteringOn = false;
        LOG_WARN("F5Low (Vulkan): flip metering refused {} times in a row, off for this session (CPU spacing)",
                 kMeterRefusals);
    }
}

// Waits (bounded) for the present `bound` frames back to be on screen before the game acquires for the next one. False:
// the game is polling (timeout 0) and it is not there yet.
bool F5Low_Vk::GateAcquire(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeoutNs)
{
    uint64_t id = 0;

    {
        std::lock_guard gateLock(_gateMutex);

        if (!_gate.on)
            return true;

        _gateBound = GateBound();
        const uint64_t number = GateWaitNumber(_issuedCount, _gateBound);

        if (number == 0 || _issuedCount - number >= kGateRing)
            return true;

        id = _issuedIds[(number - 1) % kGateRing];
    }

    if (id == 0)
        return true;

    const uint64_t waitNs = timeoutNs == 0 ? 0 : std::min<uint64_t>(timeoutNs, 100'000'000ull);
    const double waitStartMs = Util::MillisecondsNow();
    const auto result = WaitForPresent(device, swapchain, id, waitNs);
    AddAtomic(_gameWaitMs, Util::MillisecondsNow() - waitStartMs);
    AddOurWait(Util::MillisecondsNow() - waitStartMs);

    {
        std::lock_guard statLock(_gateMutex);
        ++_gateWaits;
        _gateWaitTotalMs += Util::MillisecondsNow() - waitStartMs;
    }

    // A polling game is not held and nothing is counted
    if (timeoutNs == 0 && result == VK_TIMEOUT)
        return false;

    const auto outcome = result == VK_TIMEOUT ? WaitOutcome::Timeout
                         : (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR)
                             ? WaitOutcome::Neutral
                             : WaitOutcome::Done;

    std::lock_guard gateLock(_gateMutex);
    const bool wasOn = _gate.on;
    GateOnWait(_gate, outcome, Util::MillisecondsNow());

    if (wasOn && !_gate.on)
        LOG_WARN("F5Low (Vulkan): the present wait timed out {} times in a row, the latency bound stands aside for a "
                 "while",
                 kGateGiveUp);

    return true;
}

// Present id + present wait (the latency bound) and present metering (flip metering) are added to the game's device
// when it offers them; what the game already enabled is used as it is. Present id and present wait go on together or
// not at all.
void F5Low_Vk::PrepareDevice(VkPhysicalDevice physicalDevice, VkDeviceCreateInfo& createInfo)
{
    IFGFeature_Vk::PrepareDevice(physicalDevice, createInfo);

    // The loader's own function, not the spoofing layer's (which also lists extensions the device lacks)
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        VulkanHooks::GetInstanceProcAddr(State::Instance().VulkanInstance, "vkEnumerateDeviceExtensionProperties"));

    if (enumerate == nullptr)
        enumerate = vkEnumerateDeviceExtensionProperties;

    uint32_t count = 0;
    enumerate(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    enumerate(physicalDevice, nullptr, &count, available.data());

    const auto has = [&](const char* name)
    {
        return std::ranges::any_of(available,
                                   [&](const VkExtensionProperties& e) { return strcmp(e.extensionName, name) == 0; });
    };

    VkPhysicalDeviceProperties properties {};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    const bool idExt = has(VK_KHR_PRESENT_ID_EXTENSION_NAME);
    const bool waitExt = has(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    const bool meterExt = has(VK_NV_PRESENT_METERING_EXTENSION_NAME) && properties.vendorID == 0x10DE;

    VkPhysicalDevicePresentIdFeaturesKHR idSupported { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR };
    VkPhysicalDevicePresentWaitFeaturesKHR waitSupported {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR
    };
    VkPhysicalDevicePresentMeteringFeaturesNV meterSupported { kStructMeteringFeaturesNV };

    void* chain = nullptr;

    if (meterExt)
    {
        meterSupported.pNext = chain;
        chain = &meterSupported;
    }

    if (waitExt)
    {
        waitSupported.pNext = chain;
        chain = &waitSupported;
    }

    if (idExt)
    {
        idSupported.pNext = chain;
        chain = &idSupported;
    }

    VkPhysicalDeviceFeatures2 supported { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, chain };

    if (chain != nullptr)
        vkGetPhysicalDeviceFeatures2(physicalDevice, &supported);

    // The feature flag of each structure sits at the same offset
    constexpr size_t flagOffset = offsetof(VkPhysicalDevicePresentIdFeaturesKHR, presentId);
    static_assert(offsetof(VkPhysicalDevicePresentWaitFeaturesKHR, presentWait) == flagOffset);
    static_assert(offsetof(VkPhysicalDevicePresentMeteringFeaturesNV, presentMetering) == flagOffset);

    _presentIdFeature = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR };
    _presentWaitFeature = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR };
    _meteringFeature = { kStructMeteringFeaturesNV };
    _switchedFlags.clear();

    const auto find = [&](VkStructureType type) -> VkBaseOutStructure*
    {
        for (auto* node = (VkBaseOutStructure*) createInfo.pNext; node != nullptr; node = node->pNext)
        {
            if (node->sType == type)
                return node;
        }

        return nullptr;
    };

    const auto flagOf = [&](VkBaseOutStructure* node)
    { return reinterpret_cast<VkBool32*>(reinterpret_cast<char*>(node) + flagOffset); };

    // Whether the feature can be on: already (or writable in the game's structure), or ours can be chained
    const auto canSwitch = [&](VkStructureType type)
    {
        auto* node = find(type);
        return node == nullptr || *flagOf(node) == VK_TRUE || DlssNr::VkExt::IsWritable(flagOf(node));
    };

    // The game's structure is switched on in place (put back after the create), else ours goes in front of the chain
    const auto switchOn = [&](VkStructureType type, void* oursStruct, VkBool32* oursFlag)
    {
        if (auto* node = find(type))
        {
            if (auto* flag = flagOf(node); *flag != VK_TRUE)
            {
                *flag = VK_TRUE;
                _switchedFlags.push_back(flag);
            }

            return;
        }

        *oursFlag = VK_TRUE;
        static_cast<VkBaseOutStructure*>(oursStruct)->pNext = (VkBaseOutStructure*) createInfo.pNext;
        createInfo.pNext = oursStruct;
    };

    _presentWaitOn = idExt && waitExt && idSupported.presentId && waitSupported.presentWait &&
                     canSwitch(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR) &&
                     canSwitch(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR);

    if (_presentWaitOn)
    {
        switchOn(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR, &_presentIdFeature,
                 &_presentIdFeature.presentId);
        switchOn(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR, &_presentWaitFeature,
                 &_presentWaitFeature.presentWait);
    }

    _meteringAvailable = meterExt && meterSupported.presentMetering && canSwitch(kStructMeteringFeaturesNV);

    if (_meteringAvailable)
        switchOn(kStructMeteringFeaturesNV, &_meteringFeature, &_meteringFeature.presentMetering);

    _deviceExtensions.assign(createInfo.ppEnabledExtensionNames,
                             createInfo.ppEnabledExtensionNames + createInfo.enabledExtensionCount);

    const auto addExtension = [&](const char* name)
    {
        if (std::ranges::none_of(_deviceExtensions, [&](const char* enabled) { return strcmp(enabled, name) == 0; }))
            _deviceExtensions.push_back(name);
    };

    if (_presentWaitOn)
    {
        addExtension(VK_KHR_PRESENT_ID_EXTENSION_NAME);
        addExtension(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    }

    if (_meteringAvailable)
        addExtension(VK_NV_PRESENT_METERING_EXTENSION_NAME);

    createInfo.enabledExtensionCount = static_cast<uint32_t>(_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = _deviceExtensions.data();

    LOG_INFO("F5Low (Vulkan): device extras: present wait {}, present metering {}", _presentWaitOn ? "on" : "off",
             _meteringAvailable ? "on" : "off");
}

void F5Low_Vk::OnDeviceCreated(VkDevice device)
{
    IFGFeature_Vk::OnDeviceCreated(device);

    // The game's own feature structures are as it wrote them again (the device has the features either way)
    for (auto* flag : _switchedFlags)
        *flag = VK_FALSE;

    _switchedFlags.clear();

    // The device may have been made without them after all (another layer, a refused feature)
    if (_presentWaitOn && vkGetDeviceProcAddr(device, "vkWaitForPresentKHR") == nullptr)
    {
        _presentWaitOn = false;
        LOG_WARN("F5Low (Vulkan): the device has no vkWaitForPresentKHR, no latency bound");
    }
}

void F5Low_Vk::ReleaseAll(const char* why)
{
    auto& s = *_impl;

    // At exit nothing is waited for or destroyed
    if (State::Instance().isShuttingDown)
    {
        StopWorker(false);
        return;
    }

    // A worker that does not go idle still uses all of it: left running, nothing is destroyed
    if (!QuiesceWorker(2000))
    {
        StopWorker(false);
        return;
    }

    // The watchers are joined: their stop event and the events of waits that ended early are not needed any more
    if (!s.watcher[0].joinable() && !s.watcher[1].joinable())
        s.CloseWatchHandles();

    if (s.pacer != nullptr)
    {
        s.pacerDestroy(s.pacer);
        s.pacer = nullptr;
        _timer = {};
    }

    if (s.context == nullptr && !s.slotsReady && !s.d3dReady)
        return;

    LOG_INFO("F5Low (Vulkan): releasing ({})", why);

    if (!s.deviceGone && _device != VK_NULL_HANDLE)
    {
        WaitOwnWork();

        if (s.outQueue != VK_NULL_HANDLE && s.outQueue != _gameQueue)
            vkQueueWaitIdle(s.outQueue);
    }

    if (s.context != nullptr && s.destroy != nullptr)
        s.destroy(s.context);

    s.context = nullptr;

    if (s.pacer != nullptr)
    {
        s.pacerDestroy(s.pacer);
        s.pacer = nullptr;
        _timer = {};
    }

    if (!s.deviceGone && _device != VK_NULL_HANDLE)
    {
        for (auto& slot : s.slots)
        {
            for (auto fence : slot.fences)
            {
                if (fence != VK_NULL_HANDLE)
                    vkDestroyFence(_device, fence, nullptr);
            }

            if (slot.inDone != VK_NULL_HANDLE)
                vkDestroySemaphore(_device, slot.inDone, nullptr);

            for (uint32_t k = 0; k < kMaxGenerated; k++)
            {
                if (slot.acquired[k] != VK_NULL_HANDLE)
                    vkDestroySemaphore(_device, slot.acquired[k], nullptr);

                if (slot.outDone[k] != VK_NULL_HANDLE)
                    vkDestroySemaphore(_device, slot.outDone[k], nullptr);
            }

            slot = {};
        }

        if (s.poolIn != VK_NULL_HANDLE)
            vkDestroyCommandPool(_device, s.poolIn, nullptr);

        if (s.poolOut != VK_NULL_HANDLE)
            vkDestroyCommandPool(_device, s.poolOut, nullptr);
    }

    s.poolIn = VK_NULL_HANDLE;
    s.poolOut = VK_NULL_HANDLE;
    s.slotsReady = false;
    s.cursor = 0;
    s.nextSet = 0;
    s.setSlot[0] = s.setSlot[1] = -1;
    s.outQueue = VK_NULL_HANDLE;
    s.ownQueue = false;

    s.color.Reset();
    s.hudless.Reset();
    s.motion.Reset();
    s.depthImage.Reset();

    for (auto& outSet : s.out)
    {
        for (auto& image : outSet)
            image.Reset();
    }

    s.depthBuffer.Reset();
    s.depthTexture.Reset();
    s.depthSource = VK_FORMAT_UNDEFINED;
    s.depthWidth = s.depthHeight = 0;
    s.ready.Reset();
    s.consumed.Reset();
    s.done.Reset();
    s.depthDone.Reset();
    s.depthCounter = {};
    s.list.Reset();

    for (auto& allocator : s.allocators)
        allocator.Reset();

    for (auto& value : s.allocatorValue)
        value = 0;

    s.queue12.Reset();
    s.device12.Reset();
    s.d3dReady = false;
    s.warnedNotReady = false;
}

void F5Low_Vk::DestroyFGContext()
{
    Deactivate();
    ReleaseAll("frame generation context destroyed");
}

// The swapchain goes with the game's vkDestroySwapchainKHR
bool F5Low_Vk::ReleaseSwapchain(HWND hwnd)
{
    Deactivate();
    return true;
}

bool F5Low_Vk::Shutdown()
{
    Deactivate();
    ReleaseAll("shutdown");
    return true;
}

void F5Low_Vk::OnDeviceDestroying(VkDevice device)
{
    if (device == VK_NULL_HANDLE || device != _device)
        return;

    Deactivate();
    ReleaseAll("the game is destroying its device");
    _impl->deviceGone = true;
    _impl->images.clear();
    _swapchain = VK_NULL_HANDLE;
    _headroom = 0;
}
