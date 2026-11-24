// Not built with the precompiled header: no game headers in here.
#include "GuideRunner.h"

namespace native
{

GuideRunner::~GuideRunner()
{
    for (auto& allocator : _allocators)
        if (allocator != nullptr)
            allocator->Release();

    if (_list != nullptr)
        _list->Release();
    if (_fence != nullptr)
        _fence->Release();
    if (_event != nullptr)
        CloseHandle(_event);
    if (_lastQueue != nullptr)
        _lastQueue->Release();
}

bool GuideRunner::WaitFor(UINT64 value)
{
    if (_fence == nullptr || value == 0)
        return true;

    const UINT64 completed = _fence->GetCompletedValue();

    if (completed == UINT64_MAX)
        return false; // a removed device's fence reads UINT64_MAX: nothing it reports as done was really done

    if (completed >= value)
        return true;

    ResetEvent(_event);

    if (FAILED(_fence->SetEventOnCompletion(value, _event)) || WaitForSingleObject(_event, 1000) != WAIT_OBJECT_0)
        return false;

    const UINT64 after = _fence->GetCompletedValue();
    return after != UINT64_MAX && after >= value;
}

void GuideRunner::WaitIdle()
{
    if (_fence != nullptr && _event != nullptr)
    {
        ResetEvent(_event);

        if (_fence->GetCompletedValue() != UINT64_MAX && _fence->GetCompletedValue() < _signalled &&
            SUCCEEDED(_fence->SetEventOnCompletion(_signalled, _event)))
            WaitForSingleObject(_event, 5000);
    }
}

bool GuideRunner::Init(ID3D12Device* device)
{
    for (auto& allocator : _allocators)
    {
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))))
        {
            _error = "creating a command allocator";
            return false;
        }
    }

    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, _allocators[0], nullptr,
                                         IID_PPV_ARGS(&_list))))
    {
        _error = "creating the command list";
        return false;
    }

    _list->SetName(L"Finished Picture (Vulkan game) guides");
    _list->Close();

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_fence))))
    {
        _error = "creating the fence";
        return false;
    }

    _event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    if (_event == nullptr)
    {
        _error = "creating the fence event";
        return false;
    }

    _device = device;
    return true;
}

bool GuideRunner::RunEmpty(ID3D12CommandQueue* queue, const std::function<void(ID3D12GraphicsCommandList*)>& record)
{
    if (!Ready() || queue == nullptr || !record)
        return false;

    if (_lastQueue != nullptr && queue != _lastQueue)
    {
        const UINT64 completed = _fence->GetCompletedValue();

        if (completed == UINT64_MAX || completed < _signalled)
            return false;
    }

    const UINT slot = (UINT) (_frame % kRing);

    if (!WaitFor(_values[slot]) || FAILED(_allocators[slot]->Reset()) ||
        FAILED(_list->Reset(_allocators[slot], nullptr)))
        return false;

    record(_list);

    // A list that cannot be closed is never executed: what `record` parked for it (DlssNr::ReleaseFinishedIdle watches
    // the list) is then freed by the retire list's own last-resort timer, not by this list finishing. Closing an empty
    // list does not fail in practice.
    if (FAILED(_list->Close()))
    {
        _list->Release();
        _list = nullptr;

        if (SUCCEEDED(_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, _allocators[slot], nullptr,
                                                 IID_PPV_ARGS(&_list))))
            _list->Close();

        return false;
    }

    if (queue != _lastQueue)
    {
        queue->AddRef();

        if (_lastQueue != nullptr)
            _lastQueue->Release();

        _lastQueue = queue;
    }

    ID3D12CommandList* lists[] = { _list };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(_fence, ++_signalled);
    _values[slot] = _signalled;
    ++_frame;
    return true;
}

GuideRunner::Result GuideRunner::Run(ID3D12CommandQueue* queue, const FrameInput& input, const ApplyFn& apply,
                                     FrameOutput& output)
{
    Result result;
    output = FrameOutput {};

    if (!Ready() || queue == nullptr || input.picture == nullptr || !apply)
    {
        result.stoppedAt = !Ready()           ? "not ready"
                           : queue == nullptr ? "no queue"
                           : !apply           ? "no consumer"
                                              : "no picture";
        return result;
    }

    // Another queue than the last list's: the consumer's scratch images are the same, and only lists on one queue are
    // ordered by submission. As NativeProducer does, the order is kept on the CPU, never with a GPU wait across queues.
    if (_lastQueue != nullptr && queue != _lastQueue)
    {
        const UINT64 completed = _fence->GetCompletedValue();

        if (completed == UINT64_MAX || completed < _signalled)
        {
            result.stoppedAt = "waiting for the last list on another queue";
            return result;
        }
    }

    // The slot's allocator is reset below: while the GPU may still run its last list that would break that list
    const UINT slot = (UINT) (_frame % kRing);

    if (!WaitFor(_values[slot]))
    {
        result.stoppedAt = "waiting for the allocator's last list";
        result.stoppedHr = HRESULT_FROM_WIN32(WAIT_TIMEOUT);
        return result;
    }

    const HRESULT allocatorHr = _allocators[slot]->Reset();
    const HRESULT listHr = FAILED(allocatorHr) ? allocatorHr : _list->Reset(_allocators[slot], nullptr);

    if (FAILED(listHr))
    {
        result.stoppedAt = FAILED(allocatorHr) ? "allocator reset" : "list reset";
        result.stoppedHr = (long) listHr;
        return result;
    }

    result.ran = apply(_list, input);
    const HRESULT closeHr = _list->Close();

    if (FAILED(closeHr))
    {
        result.ran = false;
        result.stoppedAt = "close";
        result.stoppedHr = (long) closeHr;

        // A list whose recording failed stays open, and every later Reset of it fails: make a fresh one
        _list->Release();
        _list = nullptr;

        if (SUCCEEDED(_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, _allocators[slot], nullptr,
                                                 IID_PPV_ARGS(&_list))))
            _list->Close();

        return result;
    }

    if (input.ready.fence != nullptr)
        queue->Wait(input.ready.fence, input.ready.value);

    if (queue != _lastQueue)
    {
        queue->AddRef();

        if (_lastQueue != nullptr)
            _lastQueue->Release();

        _lastQueue = queue;
    }

    ID3D12CommandList* lists[] = { _list };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(_fence, ++_signalled);
    _values[slot] = _signalled;
    ++_frame;

    output.done = SyncPoint { _fence, _signalled };
    output.processed = result.ran;
    result.submitted = true;
    return result;
}

} // namespace native
