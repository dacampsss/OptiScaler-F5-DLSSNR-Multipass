#pragma once

// The D3D12 half of Finished Picture on a native Vulkan game (native/VkFrameSource.h, dlssnr/DlssNr_FinishedVk.h): one
// command list on the private D3D12 device that waits for the copy in, runs the consumer (DLSS-NR on the shared
// picture, with the depth and motion the game's own upscaler call was given) and signals when the picture may be copied
// back. The producer of Optical F5Low (native/NativeProducer.h) does the same around its optical flow; here the game
// supplies the guides, so there is no flow, trust mask or depth finder, and none of their cost.
//
// Not built with the precompiled header: no game headers in here.

#include "FrameContract.h"

#include <functional>
#include <string>

namespace native
{

class GuideRunner
{
  public:
    // Records the consumer onto `cmd`, which is a list of this runner's own. True when NR ran on the picture.
    using ApplyFn = std::function<bool(ID3D12GraphicsCommandList* cmd, const FrameInput& input)>;

    struct Result
    {
        bool submitted = false;          // the work was sent: output.done is valid
        bool ran = false;                // the consumer ran on the picture
        const char* stoppedAt = nullptr; // when nothing was sent: the step that gave up (for the log)
        long stoppedHr = 0;
    };

    GuideRunner() = default;
    ~GuideRunner();
    GuideRunner(const GuideRunner&) = delete;
    GuideRunner& operator=(const GuideRunner&) = delete;

    // The command objects, on `device`. False with a reason in Error().
    bool Init(ID3D12Device* device);
    bool Ready() const { return _list != nullptr; }
    ID3D12Device* Device() const { return _device; }
    const std::string& Error() const { return _error; }

    // One frame, on `queue` (a direct queue of Device()): waits for input.ready, records, sends, and gives the point
    // the adapter waits on in output.done.
    Result Run(ID3D12CommandQueue* queue, const FrameInput& input, const ApplyFn& apply, FrameOutput& output);

    // A list of the runner's own with no picture, for work that must be ordered on the queue with its frames and
    // watched by its lists (letting go of the model while Finished Picture is idle). `record` is called with the open
    // list.
    bool RunEmpty(ID3D12CommandQueue* queue, const std::function<void(ID3D12GraphicsCommandList*)>& record);

    // Waits (at most a few seconds) for the last frame the runner sent.
    void WaitIdle();

  private:
    static constexpr UINT kRing = 3;

    bool WaitFor(UINT64 value);

    ID3D12Device* _device = nullptr;
    ID3D12CommandAllocator* _allocators[kRing] = {};
    ID3D12GraphicsCommandList* _list = nullptr;
    ID3D12Fence* _fence = nullptr;
    HANDLE _event = nullptr;
    UINT64 _values[kRing] = {};
    UINT64 _signalled = 0;
    uint64_t _frame = 0;
    ID3D12CommandQueue* _lastQueue = nullptr; // held so its address is not reused by another queue
    std::string _error;
};

} // namespace native
