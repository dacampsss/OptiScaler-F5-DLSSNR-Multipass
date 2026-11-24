#pragma once
#include "SysUtils.h"
#include <framegen/IFGFeature_Vk.h>

#include "F5LowVkRules.h"

#include <framegen/xefg/XeFG_DynamicCount.h>
#include <framegen/xefg/XeFG_ShownRate.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

// FG Output F5Low (XeFG Tensor) on a native Vulkan game's own swapchain: no bridge, no D3D12 swapchain. The game keeps
// the driver's VkSwapchain (made with spare images and copy usage); on each present the game's picture, motion and
// depth are copied into images shared with a private D3D12 device, f5low.dll's "frames only" API (f5low_frames.h) makes
// the generated frames there, and they are copied back into spare swapchain images and presented before the game's own
// image. The copy in and f5low's call run on the game thread (game queue); the presents run on a worker thread on one
// extra present-capable queue (reserved in the constructor), paced by the measured game frame interval. Without that
// queue the presents and their pacing run on the game thread as before.
class F5Low_Vk : public virtual IFGFeature_Vk
{
  public:
    struct Impl;
    struct Job;
    struct PacerCtx;

  private:
    std::unique_ptr<Impl> _impl;

    // Why the output cannot run at all (f5low.dll too old, another generator): fixed after the constructor
    std::string _blocked;
    // Why it is not generating now (the last thing that stopped it), shown in the menu; empty when it is fine
    std::string _reason;
    mutable std::mutex _reasonMutex;

    uint32_t _headroom = 0; // swapchain images beyond the game's own
    bool _hdr10 = false;
    VkColorSpaceKHR _colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    // What Present() decided for the game's present that follows
    struct Pending
    {
        bool valid = false;
        int index = -1;
        UINT64 frame = 0;
        std::optional<VkFGResource> depth, velocity, hudless;
    } _pending;

    std::atomic<double> _intervalMs { 0.0 }; // the game's own frame time (the game thread writes, the worker reads)
    double _lastEntryMs = 0.0;
    uint64_t _generatedFrames = 0;
    int _failures = 0;
    int _wanted = 1;           // generated frames per game frame the multiplier asks for
    int _countLimit = 5;       // lowered when f5low refuses a larger count
    std::string _nameCache;    // built once in the constructor
    bool _removed = false;     // the D3D12 device was removed: stays so across swapchains
    bool _retiredOld = false;  // a failed create retired the game's old swapchain
    int _createdWanted = 1;    // the multiplier the swapchain was made for
    F5LowVk::RecreateAsk _ask; // getting the game to make a new swapchain after a raise
    std::mutex _askMutex;
    VkResult AcquireLoop(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                         VkFence fence, uint32_t* pImageIndex);
    VkResult AskRecreateIfRaised(VkResult result);
    bool _loggedBehind = false;

    // Dynamic MFG: the count follows the frame rate (the setting is the most); fed with what the worker and the direct
    // presents show (xefg::AddShownPresent) and with the time the game waits in OptiScaler
    XeFG_DynamicCount _dynamicCount;
    xefg::ShownPerFrame _shownPerFrame;
    std::atomic<uint64_t> _shownTotal { 0 };
    std::atomic<double> _gameWaitMs {
        0.0
    }; // waits of the game thread in our present and acquire (the gate, the drain)
    static void AddAtomic(std::atomic<double>& total, double ms)
    {
        double seen = total.load();

        while (!total.compare_exchange_weak(seen, seen + ms))
        {
        }
    }
    bool _dynamicActive = false;
    bool _dynamicDrain = true;
    bool _dynamicUnfocused = false;
    bool _dmfgHold = false; // chose 1X
    int _dynamicKept = 0;
    int _shownCount = -1;
    int _refreshRate = 0;
    double _refreshCheckedMs = 0.0;
    double _dynamicLogMs = 0.0;
    HWND _hwnd = nullptr;
    int _hudlessMode = -1;
    uint32_t _hudlessLogs = 0;
    uint32_t _inlineGenerated = 0;
    bool _warnedCapMode = false;

    // f5low's pacer (f5low_pacer.h), when the f5low.dll has it
    std::mutex _statMutex;
    F5LowVk::SpacingStats _spacing;
    int64_t _lastPresentTicks = 0;
    uint64_t _statGenerated = 0;
    uint64_t _statBatches = 0;
    uint32_t _statFrames = 0;
    uint64_t _gateWaits = 0;
    double _gateWaitTotalMs = 0.0;
    int64_t _arriveTicks = 0;               // the real QPC of the game's present call (as on D3D12)
    std::atomic<double> _ourWaitMs { 0.0 }; // our blocking waits on the game thread since the last arrival
    // Stats window (reset every line)
    std::atomic<uint64_t> _statShown { 0 }; // every image we presented (worker, direct, inline), counted in PresentOn
    uint64_t _statShownFirst = 0, _statShownLast = 0; // _statShown at the window's first and last present
    uint64_t _statMeterFirst = 0, _statMeterLast = 0; // the overlay meter's count at the same two moments
    bool _statMeterOn = false;
    double _statFirstEntryMs = 0.0;
    double _statLastEntryMs = 0.0;
    uint32_t _statEntries = 0;   // game presents seen in the window
    uint32_t _statOffFrames = 0; // of them, frames that went out without generating
    double _statWaitMs = 0.0;    // our waits on the game thread in the window
    void AddOurWait(double ms) { AddAtomic(_ourWaitMs, ms); }

    // Device extensions this output enabled (or found enabled): present id + present wait bound how far the game runs
    // ahead, present metering lets the RTX 50 display hardware space the flips of a batch
    bool _presentWaitOn = false;
    bool _meteringAvailable = false;
    std::atomic<bool> _meteringOn { false };
    std::atomic<bool> _meterWasOn { false }; // the last batch was metered: the next present that is not closes it
    std::atomic<bool> _gateEnabled { false };
    uint32_t _gateBound = 0;
    std::atomic<uint32_t> _gateCount { 1 }; // generated frames per game frame the bound is for (1 = 2X)
    std::atomic<int> _gateMode { 0 };       // the present mode of the last frame (its effective one)
    uint32_t GateBound() const
    {
        return F5LowVk::LatencyBound(_gameMinImages, _gateCount.load(), F5LowVk::VsyncPaces(_gateMode.load()));
    }
    int _presentMode = 0;     // the mode our swapchain was made with
    int _gamePresentMode = 0; // the mode the game asked for
    bool _tearing = false;    // MAILBOX asked, IMMEDIATE made (VulkanTearing)
    std::mutex _gateMutex;    // _gate, the ids, _meterHealth
    F5LowVk::GateState _gate;
    F5LowVk::MeterHealth _meterHealth;
    uint64_t _ownPresentId = 0;
    uint64_t _issuedIds[F5LowVk::kGateRing] {};
    uint64_t _issuedCount = 0;
    uint64_t _frameOwnId = 0; // the id this frame's present carries when the game chained none
    uint32_t _inlineBatch =
        0; // presented generated frames + 1 of a frame whose generated frames ran inline (no worker)
    uint32_t _gameMinImages = 0;
    uint32_t _meterLogs = 0;
    std::atomic<uint32_t> _lastFeedback { 0 };
    bool _hostMeter = false; // the pacer was told the host can meter (ini + device)
    F5LowVk::TimerState _timer;
    std::vector<VkBool32*>
        _switchedFlags; // feature flags switched on in the game's own structs, put back after the create
    std::vector<const char*> _deviceExtensions;
    VkPhysicalDevicePresentIdFeaturesKHR _presentIdFeature {};
    VkPhysicalDevicePresentWaitFeaturesKHR _presentWaitFeature {};
    VkPhysicalDevicePresentMeteringFeaturesNV _meteringFeature {};
    bool DecideMetering();
    void NoteIssued(uint64_t id);
    bool GateAcquire(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeoutNs);
    void NoteMeterFeedback(VkResult result, uint32_t feedback);
    bool _loggedGenerating = false; // the first generated frame of this swapchain was logged

    void SetReason(const std::string& reason);
    void UpdateCount(bool perFrame = false);
    int DynamicCount(int maxCount);
    float DisplayRefreshFps(double nowMs);

    bool Prepare(const Pending& pending, uint32_t count, bool& useHudless, uint32_t set);
    bool EnsureSlots();

    // Our own GPU work and the worker, so a size change or a teardown can go on: the worker drained, every fence of
    // ours waited
    void WaitOwnWork();
    bool DrainWorker(DWORD timeoutMs);
    bool QuiesceWorker(DWORD timeoutMs);
    void LatchOff(const std::string& reason);
    bool CarryPresentStructs(const VkPresentInfoKHR* present, Job& job);
    void StartWorker();
    void StopWorker(bool join);
    void WorkerLoop();

    // The generated frames of a job: acquire, copy, present, spaced. On the worker, or on the game thread without a
    // present queue. Returns after the last generated frame was presented.
    uint32_t RunGenerated(const Job& job, bool paced, PacerCtx* pacer = nullptr);
    bool PacerStart(PacerCtx& pc, uint32_t k, bool generatedFrame);
    void PacerPresented(PacerCtx& pc, const Job& job, uint32_t k, bool dropped, VkResult result, uint32_t feedback);
    bool NextGameFrameDone();
    void EnsurePacer();
    int EffectiveMode(const Job& job) const;
    void LogStats();
    bool AcquireImage(VkSemaphore semaphore, uint32_t* index, double timeoutMs);
    VkResult PresentOn(VkQueue queue, const VkPresentInfoKHR& info, DWORD lockTimeoutMs);
    VkResult DirectPresent(VkQueue queue, const VkPresentInfoKHR* info, uint64_t ownId = 0, uint32_t meterBatch = 0);

    enum class Outcome
    {
        Direct, // the real present goes out on the game thread (finalInfo)
        Queued  // the worker presents everything
    };
    Outcome Generate(VkQueue queue, const VkPresentInfoKHR* present, Pending& pending, VkPresentInfoKHR& finalInfo);
    void ReleaseAll(const char* why);

  public:
    // IFGFeature
    feature_version Version() override final;
    const char* Name() override final;
    bool Present() override final;
    void Activate() override final;
    void Deactivate() override final;
    void DestroyFGContext() override final;
    bool ReleaseSwapchain(HWND hwnd) override final;
    bool Shutdown() override final;
    bool SetInterpolatedFrameCount(UINT interpolatedFrameCount) override final { return true; }

    // IFGFeature_Vk: the game's handle is the driver's, VK_NV_low_latency2 calls with it reach the driver
    bool HasDriverSwapchain() const override final { return true; }
    bool WantsLatencyMode() const override final { return true; }
    void OnDeviceDestroying(VkDevice device) override final;
    void PrepareDevice(VkPhysicalDevice physicalDevice, VkDeviceCreateInfo& createInfo) override final;
    void OnDeviceCreated(VkDevice device) override final;
    bool RetiredOldSwapchain() const override final { return _retiredOld; }
    VkResult WaitForPresent(VkDevice device, VkSwapchainKHR swapchain, uint64_t presentId,
                            uint64_t timeout) override final;

    bool CreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
                         const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) override final;
    bool DestroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                          const VkAllocationCallbacks* pAllocator) override final;
    VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount,
                                VkImage* pSwapchainImages) override final;
    VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                              VkFence fence, uint32_t* pImageIndex) override final;
    VkResult QueuePresent(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) override final;
    void SetHdrMetadata(VkDevice device, VkSwapchainKHR swapchain, const VkHdrMetadataEXT* pMetadata) override final;

    void EvaluateState(FG_Constants& fgConstants) override final;

    std::string StatusLine() const override final;
    bool DynamicHold() const override final { return _dmfgHold && _dynamicActive; }

    F5Low_Vk();
    ~F5Low_Vk();
};
