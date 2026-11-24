#include "pch.h"
#include "XeSS_UE_Dx12.h"
#include "XeSS_UE_Decode.h"

#include <Config.h>
#include <State.h>

#include <chrono>
#include <psapi.h>
#include <detours/detours.h>
#include <scanner/scanner.h>
#include <shaders/depth_scale/DS_Dx12.h>

// Ported from upstream OptiScaler 7e941561 (FakeMichau, GPL-3.0). Differences from upstream: references the GPU can
// still use go through a retire list of its own (this fork has no DeferredRelease), growing the copy texture retires
// the old one instead of freeing it under frames in flight, copies stop for a device other than the first, a failed
// hook install is undone, [XeSS] UnrealDepth turns it off, and the instruction decoding is in XeSS_UE_Decode.h, host
// tested.

namespace
{
// Most plugin versions have a function shaped like
// FRDGTextureRef (FRDGBuilder&, FRDGTextureRef SceneDepth, FRDGTextureRef SceneVelocity, const FViewInfo&)
// that names the flattened velocity texture. The stack arguments are passed through in case a build has more.
using PFN_AddVelocityFlatteningXeSSPass = void*(__fastcall*) (void* graphBuilder, void* sceneDepth, void* sceneVelocity,
                                                              void* view, void* arg5, void* arg6, void* arg7,
                                                              void* arg8);
PFN_AddVelocityFlatteningXeSSPass o_AddVelocityFlatteningXeSSPass = nullptr;

// Builds that inline the XeSS wrapper into the upscaler's AddPasses call the flattening it shares with the XeFG plugin
// directly: FRDGTextureRef (FRDGBuilder&, FRDGTextureRef SceneDepth, FRDGTextureRef SceneVelocity, const FViewInfo&,
// const TCHAR* Name, event, bool bUpsample). bUpsample sizes its output at the display size, without it the output
// is at View.ViewRect like XeFG's, which gives render resolution motion vectors.
using PFN_FlattenVelocity = void*(__fastcall*) (void* graphBuilder, void* sceneDepth, void* sceneVelocity, void* view,
                                                const wchar_t* name, void* event, bool upsample);
PFN_FlattenVelocity o_FlattenVelocity = nullptr;
const wchar_t* _xessVelocityName = nullptr; // XeSS's calls name their texture with it, XeFG's are left alone

// The flattening sizes its output with FViewInfo::GetSecondaryViewRectSize(), the display size. While it runs, that
// size is replaced with View.ViewRect's so the motion vectors come out at render resolution, which XeFG needs.
using PFN_GetSecondaryViewRectSize = int32_t*(__fastcall*) (void* view, int32_t* outSize, void* r8, void* r9);
PFN_GetSecondaryViewRectSize o_GetSecondaryViewRectSize = nullptr;
int32_t _viewRectOffset = 0;
thread_local void* t_flatteningView = nullptr;

// Scene depth FRDGTexture from the latest flattening, and its name pointer to recognise it later. UE 4 registers
// scene depth as an external texture with its FRHITexture already set during graph setup, UE 5 creates it in the
// graph and it only gets one once the graph executes.
std::atomic<void*> _sceneDepthRdg { nullptr };
std::atomic<const wchar_t*> _sceneDepthName { nullptr };

// FRHITexture of the scene depth, a pooled render target that outlives the frames using it
std::atomic<void*> _sceneDepthRhi { nullptr };

// Present count of the latest flattening. The FRDGTexture is in graph memory reused after its frame, and the engine
// releases the FRHITexture once it stops using the scene depth, during a load for example, so they are only read for
// a few presents after a flattening.
std::atomic<uint64_t> _flattenedAt { 0 };
constexpr uint64_t ProbePresents = 8;

// What a scene depth has to match: the vtable of an ID3D12Resource and the extent of the scene color's view rect from
// the latest execute, and the format of a confirmed scene depth
std::atomic<uintptr_t> _resourceVtable { 0 };
std::atomic<uint32_t> _minWidth { 0 };
std::atomic<uint32_t> _minHeight { 0 };
std::atomic<uint32_t> _depthFormat { DXGI_FORMAT_UNKNOWN };

size_t _resolvedOffsets[2] = {};

constexpr size_t MaxCandidates = 12;

// One per descriptor heap of the shader, a copy list is reused only once the GPU is done with it and its heap
constexpr size_t CopyListCount = DS_NUM_OF_HEAPS;

// Requests a confirmation stays valid for. A texture the engine stopped using as the depth returns to its pool and
// can be written as another depth target.
constexpr uint64_t ConfirmedRequests = 120;

// Requests since the latest copy, past this the copy no longer follows the frames
constexpr uint32_t StaleRequests = 60;

struct QueueState
{
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    uint64_t written = 0; // queue order of the latest transition to DEPTH_WRITE
};

struct Transition
{
    ID3D12Resource* resource;
    D3D12_RESOURCE_STATES state;
};

struct CandidateInfo
{
    bool confirmed = false;   // found through the FRDGTexture, not only by its description
    uint64_t confirmedAt = 0; // request count of the latest confirmation
    uint64_t lastUsed = 0;    // write order of the latest write or confirmation, for eviction
};

// Read without the lock by the barrier hook, which runs for every barrier the game records
std::array<std::atomic<ID3D12Resource*>, MaxCandidates> _candidates {};
std::atomic<uint32_t> _candidateCount { 0 };

// UE 5 alternates between two scene depth textures and the one an FRDGTexture holds when XeSS runs on the RHI thread
// can already belong to the next frame, whose depth is still being written. Command lists are recorded in parallel, so
// which one is this frame's and its state are only known once they are submitted. Transitions on the candidates are
// recorded per command list and replayed in queue order at submission, and right before the command list that ran
// XeSS a copy of the candidate written since the previous copy is submitted. The engine's textures are never
// transitioned.
//
// Recursive because the copy list is recorded with the lock held and its own Reset and barriers go through the hooks
std::recursive_mutex _mutex;

std::array<CandidateInfo, MaxCandidates> _candidateInfo {};
std::unordered_map<ID3D12CommandList*, std::vector<Transition>> _recorded;
std::unordered_map<ID3D12Resource*, QueueState> _queueStates;
uint64_t _writeOrder = 0;
uint64_t _lastCopyPoint = 0; // write order where the previous copy was due
std::unordered_set<ID3D12CommandList*> _requests;
uint64_t _requestCount = 0;
uint32_t _requestsSinceCopy = 0;

// References the GPU can still be using: replaced candidates and outgrown copy textures. Released after RetirePresents
// presents and RetireAge, or after RetireFallbackAge when presents aren't counted (no wrapped swap chain).
struct Retired
{
    ID3D12Resource* resource;
    uint64_t present;
    std::chrono::steady_clock::time_point time;
};

constexpr uint64_t RetirePresents = 8;
constexpr auto RetireAge = std::chrono::milliseconds(500);
constexpr auto RetireFallbackAge = std::chrono::seconds(2);
std::vector<Retired> _retired;

// The device the copy objects belong to, kept without a reference: the game's device outlives the copy
ID3D12Device* _device = nullptr;
DS_Dx12* _shader = nullptr;
std::array<ID3D12CommandAllocator*, CopyListCount> _allocators {};
std::array<ID3D12GraphicsCommandList*, CopyListCount> _copyLists {};
// A fence per copy list: signals from different queues aren't ordered, so a shared fence could show a list as done
// while it still runs on another queue
std::array<ID3D12Fence*, CopyListCount> _fences {};
std::array<uint64_t, CopyListCount> _copyFenceValues {};
uint32_t _nextCopyList = 0;

using PFN_ResourceBarrier = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList* This, UINT NumBarriers,
                                                     const D3D12_RESOURCE_BARRIER* pBarriers);
using PFN_CommandListReset = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList* This,
                                                         ID3D12CommandAllocator* pAllocator,
                                                         ID3D12PipelineState* pInitialState);
using PFN_ExecuteCommandLists = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue* This, UINT NumCommandLists,
                                                         ID3D12CommandList* const* ppCommandLists);

PFN_ResourceBarrier o_ResourceBarrier = nullptr;
PFN_CommandListReset o_CommandListReset = nullptr;
PFN_ExecuteCommandLists o_ExecuteCommandLists = nullptr;

bool IsUserPointer(uintptr_t value) { return value >= 0x10000 && value < 0x7FFFFFFF0000 && (value & 7) == 0; }

// Engine memory can already be freed and reused
template <typename T> bool SafeRead(uintptr_t address, T& out)
{
    __try
    {
        memcpy(&out, (const void*) address, sizeof(T));
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return false;
    }
}

bool SafeGetDesc(ID3D12Resource* resource, D3D12_RESOURCE_DESC& desc)
{
    __try
    {
        desc = resource->GetDesc();
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return false;
    }
}

// The caller holds the lock
void RetireLocked(ID3D12Resource* resource)
{
    if (resource != nullptr)
        _retired.push_back({ resource, State::Instance().frameCount, std::chrono::steady_clock::now() });
}

// The caller holds the lock
void CollectRetiredLocked()
{
    // The process is going away, the OS cleans up
    if (State::Instance().isShuttingDown)
        return;

    auto now = std::chrono::steady_clock::now();
    auto present = State::Instance().frameCount;

    std::erase_if(_retired,
                  [&](const Retired& retired)
                  {
                      auto age = now - retired.time;
                      auto ready =
                          age >= RetireFallbackAge || (present - retired.present >= RetirePresents && age >= RetireAge);

                      if (ready)
                          retired.resource->Release();

                      return ready;
                  });
}

// FRDGTexture { vtable, const TCHAR* Name, FRHIResource* ResourceRHI } in UE 4.26+ and UE 5
struct RdgTexture
{
    void* vtable;
    const wchar_t* name;
    void* rhiTexture;
};

// A resource name, which tells an FRDGTexture apart from other memory. The scene depth is usually SceneDepthZ, but
// renderers can replace it, like Borderlands 4's menus with SLW.DepthPrepassOutput.
bool ReadResourceName(const wchar_t* name, std::wstring& text)
{
    constexpr size_t MaxLength = 64;
    text.clear();

    for (size_t i = 0; i < MaxLength; i++)
    {
        wchar_t c = 0;

        if (!SafeRead((uintptr_t) (name + i), c))
            return false;

        if (c == L'\0')
            return !text.empty();

        if (c < 0x20 || c > 0x7E)
            return false;

        text.push_back(c);
    }

    return false;
}

bool IsDepthResource(uintptr_t candidate, uintptr_t resourceVtable)
{
    uintptr_t vtable = 0;
    D3D12_RESOURCE_DESC desc {};

    return IsUserPointer(candidate) && SafeRead(candidate, vtable) && vtable == resourceVtable &&
           SafeGetDesc((ID3D12Resource*) candidate, desc) && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
}

// The D3D12 RHI keeps the ID3D12Resource behind FD3D12Texture -> FD3D12ResourceLocation -> FD3D12Resource, at
// offsets that differ between engine versions and game forks. Instead of hardcoding them, look for a pointer, one or
// two levels deep, to an object with the vtable of a known ID3D12Resource and remember where it was.
ID3D12Resource* FindD3D12Resource(void* rhiTexture, uintptr_t resourceVtable)
{
    constexpr size_t TextureSearchSize = 0x400;
    constexpr size_t ResourceSearchSize = 0x80;
    auto texture = (uintptr_t) rhiTexture;
    uintptr_t level1 = 0;
    uintptr_t level2 = 0;

    // Once the offsets are known only they are used, so a pointer that isn't a texture can't match something else
    if (_resolvedOffsets[0] != 0)
    {
        if (SafeRead(texture + _resolvedOffsets[0], level1) && IsUserPointer(level1) &&
            SafeRead(level1 + _resolvedOffsets[1], level2) && IsDepthResource(level2, resourceVtable))
        {
            return (ID3D12Resource*) level2;
        }

        return nullptr;
    }

    for (size_t offset1 = 8; offset1 < TextureSearchSize; offset1 += 8)
    {
        if (!SafeRead(texture + offset1, level1) || !IsUserPointer(level1))
            continue;

        for (size_t offset2 = 0; offset2 < ResourceSearchSize; offset2 += 8)
        {
            if (!SafeRead(level1 + offset2, level2) || !IsDepthResource(level2, resourceVtable))
                continue;

            _resolvedOffsets[0] = offset1;
            _resolvedOffsets[1] = offset2;
            LOG_INFO("Scene depth ID3D12Resource found at FRHITexture +{:X} -> +{:X}", offset1, offset2);

            return (ID3D12Resource*) level2;
        }
    }

    return nullptr;
}

bool CoversInput(const D3D12_RESOURCE_DESC& desc)
{
    return desc.Width >= _minWidth.load() && desc.Height >= _minHeight.load();
}

// Another depth texture like the scene depth starting to be written, after a resolution change the engine allocates
// new ones before they can be confirmed
bool MatchesFilter(ID3D12Resource* resource)
{
    auto format = _depthFormat.load(std::memory_order_relaxed);

    if (format == DXGI_FORMAT_UNKNOWN)
        return false;

    auto desc = resource->GetDesc();

    return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.Format == format &&
           (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0 && CoversInput(desc);
}

int CandidateSlot(ID3D12Resource* resource)
{
    for (size_t i = 0; i < _candidates.size(); i++)
    {
        if (_candidates[i].load(std::memory_order_relaxed) == resource)
            return (int) i;
    }

    return -1;
}

bool IsComputeReadable(D3D12_RESOURCE_STATES state)
{
    return (state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) != 0 &&
           (state & D3D12_RESOURCE_STATE_DEPTH_WRITE) == 0;
}

// Takes the place of the candidate used longest ago, the caller holds the lock
int AddCandidateLocked(ID3D12Resource* depth)
{
    size_t slot = 0;

    for (size_t i = 0; i < _candidates.size(); i++)
    {
        if (_candidates[i].load() == nullptr)
        {
            slot = i;
            break;
        }

        if (_candidateInfo[i].lastUsed < _candidateInfo[slot].lastUsed)
            slot = i;
    }

    depth->AddRef();

    auto previous = _candidates[slot].exchange(depth);
    _candidateInfo[slot] = { false, 0, _writeOrder };

    if (previous != nullptr)
    {
        _queueStates.erase(previous);

        // A copy list in flight can still read it, and our reference can be the last one
        RetireLocked(previous);
    }
    else
    {
        _candidateCount++;
    }

    auto desc = depth->GetDesc();
    LOG_INFO("Scene depth candidate: {:X}, {}x{}, format: {}", (size_t) depth, desc.Width, desc.Height,
             (UINT) desc.Format);

    return (int) slot;
}

// A texture found through the FRDGTexture, preferred over candidates found by description
void AddConfirmedCandidate(ID3D12Resource* depth)
{
    std::scoped_lock lock(_mutex);

    auto slot = CandidateSlot(depth);

    if (slot < 0)
        slot = AddCandidateLocked(depth);

    auto& info = _candidateInfo[slot];
    info.confirmed = true;
    info.confirmedAt = _requestCount;
    info.lastUsed = std::max(info.lastUsed, _writeOrder);

    _depthFormat = depth->GetDesc().Format;
}

void STDMETHODCALLTYPE hkResourceBarrier(ID3D12GraphicsCommandList* This, UINT NumBarriers,
                                         const D3D12_RESOURCE_BARRIER* pBarriers)
{
    if (_candidateCount.load(std::memory_order_relaxed) != 0 && pBarriers != nullptr)
    {
        for (UINT i = 0; i < NumBarriers; i++)
        {
            auto& barrier = pBarriers[i];

            // Plane 0 is depth, plane 1 the stencil
            if (barrier.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION ||
                (barrier.Transition.Subresource != 0 &&
                 barrier.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES))
            {
                continue;
            }

            auto resource = barrier.Transition.pResource;
            auto known = CandidateSlot(resource) >= 0;

            if (!known &&
                ((barrier.Transition.StateAfter & D3D12_RESOURCE_STATE_DEPTH_WRITE) == 0 || !MatchesFilter(resource)))
            {
                continue;
            }

            std::scoped_lock lock(_mutex);

            if (!known && CandidateSlot(resource) < 0)
                AddCandidateLocked(resource);

            _recorded[This].push_back({ resource, barrier.Transition.StateAfter });
        }
    }

    o_ResourceBarrier(This, NumBarriers, pBarriers);
}

// A reset command list starts a new recording, what was recorded before belongs to an older frame
HRESULT STDMETHODCALLTYPE hkCommandListReset(ID3D12GraphicsCommandList* This, ID3D12CommandAllocator* pAllocator,
                                             ID3D12PipelineState* pInitialState)
{
    if (_candidateCount.load(std::memory_order_relaxed) != 0)
    {
        std::scoped_lock lock(_mutex);
        _recorded.erase(This);
        _requests.erase(This);
    }

    return o_CommandListReset(This, pAllocator, pInitialState);
}

void ApplyTransitions(ID3D12CommandList* commandList)
{
    auto recorded = _recorded.find(commandList);

    if (recorded == _recorded.end())
        return;

    for (auto& transition : recorded->second)
    {
        // Recorded before the candidate was replaced, the address can be reused by another resource
        auto slot = CandidateSlot(transition.resource);
        if (slot < 0)
            continue;

        auto& queueState = _queueStates[transition.resource];
        queueState.state = transition.state;

        if (transition.state & D3D12_RESOURCE_STATE_DEPTH_WRITE)
        {
            queueState.written = ++_writeOrder;
            _candidateInfo[slot].lastUsed = _writeOrder;
        }
    }

    _recorded.erase(recorded);
}

// The depth of this frame at the current point of the queue: a candidate written since the previous copy and now
// readable by compute. One confirmed recently comes first, otherwise the one written first, as the depth prepass
// writes the scene depth before other depth targets like custom depth.
ID3D12Resource* SelectSource()
{
    ID3D12Resource* best = nullptr;
    bool bestConfirmed = false;
    uint64_t bestWritten = UINT64_MAX;

    for (size_t i = 0; i < _candidates.size(); i++)
    {
        auto resource = _candidates[i].load();

        if (resource == nullptr)
            continue;

        auto queueState = _queueStates.find(resource);

        if (queueState == _queueStates.end() || queueState->second.written <= _lastCopyPoint ||
            !IsComputeReadable(queueState->second.state))
        {
            continue;
        }

        auto& info = _candidateInfo[i];
        auto confirmed = info.confirmed && _requestCount - info.confirmedAt <= ConfirmedRequests;

        if ((confirmed && !bestConfirmed) || (confirmed == bestConfirmed && queueState->second.written < bestWritten))
        {
            best = resource;
            bestConfirmed = confirmed;
            bestWritten = queueState->second.written;
        }
    }

    _lastCopyPoint = _writeOrder;

    return best;
}

ID3D12GraphicsCommandList* RecordCopy(ID3D12CommandQueue* queue, uint32_t& copyIndex)
{
    if (_shader == nullptr || !_shader->CanRender() || queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return nullptr;

    auto source = SelectSource();

    if (source == nullptr)
        return nullptr;

    auto index = _nextCopyList;

    // Still on the GPU, a frame without the copy is better than waiting
    if (_fences[index]->GetCompletedValue() < _copyFenceValues[index])
        return nullptr;

    auto allocator = _allocators[index];
    auto commandList = _copyLists[index];

    if (FAILED(allocator->Reset()) || FAILED(commandList->Reset(allocator, nullptr)))
        return nullptr;

    auto buffer = _shader->Buffer();
    auto sourceDesc = source->GetDesc();
    auto bufferDesc = buffer->GetDesc();
    auto width = (UINT) std::min(sourceDesc.Width, bufferDesc.Width);
    auto height = std::min(sourceDesc.Height, bufferDesc.Height);

    _shader->SetBufferState(commandList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto dispatched = _shader->Dispatch(commandList, source, buffer, 1.0f, width, height);
    _shader->SetBufferState(commandList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // The shader moved on to its next descriptor heap even if the dispatch failed, the copy lists follow it in step so
    // a heap is only rewritten once the list that used it is done
    _nextCopyList = (index + 1) % CopyListCount;

    if (FAILED(commandList->Close()) || !dispatched)
        return nullptr;

    _requestsSinceCopy = 0;
    _copyFenceValues[index]++;
    copyIndex = index;

    return commandList;
}

void ProbeSceneDepth();

void STDMETHODCALLTYPE hkExecuteCommandLists(ID3D12CommandQueue* This, UINT NumCommandLists,
                                             ID3D12CommandList* const* ppCommandLists)
{
    ProbeSceneDepth();

    if (_candidateCount.load(std::memory_order_relaxed) == 0)
        return o_ExecuteCommandLists(This, NumCommandLists, ppCommandLists);

    struct Submission
    {
        UINT start;
        UINT count;
        ID3D12GraphicsCommandList* copy;
        uint32_t copyIndex;
    };

    std::vector<Submission> submissions;

    {
        std::scoped_lock lock(_mutex);
        UINT start = 0;

        for (UINT i = 0; i < NumCommandLists; i++)
        {
            if (_requests.erase(ppCommandLists[i]) != 0)
            {
                uint32_t copyIndex = 0;

                if (auto copy = RecordCopy(This, copyIndex); copy != nullptr)
                {
                    submissions.push_back({ start, i - start, copy, copyIndex });
                    start = i;
                }
            }

            ApplyTransitions(ppCommandLists[i]);
        }

        submissions.push_back({ start, NumCommandLists - start, nullptr, 0 });
    }

    for (auto& submission : submissions)
    {
        if (submission.count != 0)
            o_ExecuteCommandLists(This, submission.count, ppCommandLists + submission.start);

        if (submission.copy != nullptr)
        {
            ID3D12CommandList* copy[] = { submission.copy };
            o_ExecuteCommandLists(This, 1, copy);
            This->Signal(_fences[submission.copyIndex], _copyFenceValues[submission.copyIndex]);
        }
    }
}

bool InitCopy(ID3D12GraphicsCommandList* commandList, ID3D12Device* device)
{
    if (_shader != nullptr)
        return _shader->IsInit();

    _device = device;
    _shader = new DS_Dx12("Scene Depth Copy", device);

    for (size_t i = 0; i < CopyListCount; i++)
    {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_fences[i]))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&_allocators[i]))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, _allocators[i], nullptr,
                                             IID_PPV_ARGS(&_copyLists[i]))))
        {
            return false;
        }

        _copyLists[i]->SetName(L"Scene Depth Copy");
        _copyLists[i]->Close();
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;

    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))))
        return false;

    auto listVtable = *(void***) commandList;
    auto queueVtable = *(void***) queue;

    o_CommandListReset = (PFN_CommandListReset) listVtable[10];
    o_ResourceBarrier = (PFN_ResourceBarrier) listVtable[26];
    o_ExecuteCommandLists = (PFN_ExecuteCommandLists) queueVtable[10];

    queue->Release();

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&) o_CommandListReset, hkCommandListReset);
    DetourAttach(&(PVOID&) o_ResourceBarrier, hkResourceBarrier);
    DetourAttach(&(PVOID&) o_ExecuteCommandLists, hkExecuteCommandLists);

    if (auto result = DetourTransactionCommit(); result != NO_ERROR)
    {
        // Requests would wait on a submission hook that never runs
        LOG_ERROR("Hooking ResourceBarrier, Reset and ExecuteCommandLists failed: {}, no scene depth copies", result);
        o_CommandListReset = nullptr;
        o_ResourceBarrier = nullptr;
        o_ExecuteCommandLists = nullptr;
        return false;
    }

    LOG_INFO("Hooked ResourceBarrier, Reset and ExecuteCommandLists to copy the scene depth in queue order");

    return _shader->IsInit();
}

// Texture with the depth for commands recorded into commandList, filled when the list is submitted. In
// D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE and covering the input extent, with layout as the template for its
// other properties. When no copy can be made for a frame it keeps the previous depth, or zeros, the far plane, right
// after it is created, which the upscaler survives, unlike no depth. nullptr once copies stopped, rather than a depth
// that no longer follows the frames.
ID3D12Resource* RequestCopy(ID3D12GraphicsCommandList* commandList, ID3D12Resource* layout)
{
    if (_candidateCount.load() == 0)
        return nullptr;

    std::scoped_lock lock(_mutex);

    CollectRetiredLocked();

    ID3D12Device* device = nullptr;

    if (FAILED(commandList->GetDevice(IID_PPV_ARGS(&device))))
        return nullptr;

    // The game's device outlives the copy, it is kept without a reference
    device->Release();

    if (_shader == nullptr)
        InitCopy(commandList, device);

    // The copy objects belong to the first device, a game that made another one gets no depth rather than a mix
    if (device != _device)
    {
        static bool logged = false;

        if (!logged)
        {
            logged = true;
            LOG_WARN("XeSS runs on another device than the scene depth copy was made for, no scene depth");
        }

        return nullptr;
    }

    if (!_shader->IsInit() || o_ExecuteCommandLists == nullptr)
        return nullptr;

    // Only grows, so resolution changes don't recreate it back and forth while frames still use it
    auto buffer = _shader->Buffer();
    uint32_t width = _minWidth.load();
    uint32_t height = _minHeight.load();

    if (buffer != nullptr)
    {
        auto desc = buffer->GetDesc();
        width = std::max(width, (uint32_t) desc.Width);
        height = std::max(height, desc.Height);
    }

    if (buffer == nullptr || buffer->GetDesc().Width != width || buffer->GetDesc().Height != height)
    {
        // The shader releases the old texture at once, frames in flight can still read it
        if (buffer != nullptr)
            buffer->AddRef();

        auto created = _shader->CreateBufferResource(_device, layout, width, height,
                                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        RetireLocked(buffer);

        if (!created)
            return nullptr;
    }

    _requests.insert(commandList);
    _requestCount++;

    if (++_requestsSinceCopy == StaleRequests)
        LOG_WARN("Scene depth not copied for {} frames, the upscaler gets no depth until it is", StaleRequests);

    return _requestsSinceCopy < StaleRequests ? _shader->Buffer() : nullptr;
}

// Looks for the scene depth texture through the latest FRDGTexture. Runs when XeSS executes and on every queue
// submission: XeSS runs from an RHI thread command, often after the graph that held the FRDGTexture is gone and its
// memory reused, while earlier submissions of a frame overlap the execution of its graph. A new scene depth, after a
// load or a resolution change, is then found in the same frame.
void ProbeSceneDepth()
{
    static std::mutex mutex;
    std::scoped_lock lock(mutex);

    auto vtable = _resourceVtable.load();

    if (vtable == 0 || State::Instance().frameCount - _flattenedAt.load() > ProbePresents)
        return;

    void* rhiTexture = nullptr;

    // The name pointer tells whether the memory can still be the scene depth and the resolve whether its
    // FRHITexture is one
    if (auto rdgTexture = _sceneDepthRdg.load(); rdgTexture != nullptr)
    {
        RdgTexture rdg {};

        if (SafeRead((uintptr_t) rdgTexture, rdg) && rdg.name == _sceneDepthName.load())
            rhiTexture = rdg.rhiTexture;
    }

    // UE 4 has it from the graph setup
    auto setupRhiTexture = _sceneDepthRhi.load();

    if (rhiTexture == nullptr)
        rhiTexture = setupRhiTexture;

    if (!IsUserPointer((uintptr_t) rhiTexture))
        return;

    auto depth = FindD3D12Resource(rhiTexture, vtable);

    if (depth == nullptr || !CoversInput(depth->GetDesc()))
    {
        // Released or reallocated by the engine, its memory can now be anything
        if (rhiTexture == setupRhiTexture)
            _sceneDepthRhi.compare_exchange_strong(setupRhiTexture, nullptr);

        return;
    }

    _sceneDepthRhi = rhiTexture;
    AddConfirmedCandidate(depth);
}

int32_t* __fastcall hkGetSecondaryViewRectSize(void* view, int32_t* outSize, void* r8, void* r9)
{
    auto result = o_GetSecondaryViewRectSize(view, outSize, r8, r9);

    if (view == nullptr || view != t_flatteningView || result == nullptr)
        return result;

    int32_t rect[4] {}; // FIntRect { Min.X, Min.Y, Max.X, Max.Y }

    if (!SafeRead((uintptr_t) view + _viewRectOffset, rect))
        return result;

    auto width = rect[2] - rect[0];
    auto height = rect[3] - rect[1];

    // ViewRect is never larger than the display size, anything else means the offset is wrong for this build
    if (rect[0] < 0 || rect[1] < 0 || width <= 0 || height <= 0 || width > result[0] || height > result[1])
    {
        static bool logged = false;

        if (!logged)
        {
            logged = true;
            LOG_WARN("Unexpected View.ViewRect: {},{} {},{} for display size {}x{}", rect[0], rect[1], rect[2], rect[3],
                     result[0], result[1]);
        }

        return result;
    }

    result[0] = width;
    result[1] = height;

    return result;
}

void RecordSceneDepth(void* sceneDepth)
{
    RdgTexture rdg {};
    std::wstring name;

    if (sceneDepth == nullptr || !SafeRead((uintptr_t) sceneDepth, rdg) || rdg.name == nullptr ||
        ((uintptr_t) rdg.name & 1) != 0 || !ReadResourceName(rdg.name, name))
    {
        static bool logged = false;

        if (!logged)
        {
            logged = true;
            LOG_WARN("Velocity flattening depth input isn't an FRDGTexture: {:X}", (size_t) sceneDepth);
        }

        return;
    }

    static std::atomic<uint32_t> nameChanges { 0 };

    if (_sceneDepthName.exchange(rdg.name) != rdg.name && nameChanges++ < 16)
        LOG_INFO("Velocity flattening depth input: {}", wstring_to_string(name));

    _sceneDepthRdg = sceneDepth;

    if (rdg.rhiTexture != nullptr)
        _sceneDepthRhi = rdg.rhiTexture;

    _flattenedAt = State::Instance().frameCount;
}

void* __fastcall hkAddVelocityFlatteningXeSSPass(void* graphBuilder, void* sceneDepth, void* sceneVelocity, void* view,
                                                 void* arg5, void* arg6, void* arg7, void* arg8)
{
    RecordSceneDepth(sceneDepth);

    t_flatteningView = o_GetSecondaryViewRectSize != nullptr ? view : nullptr;
    auto result =
        o_AddVelocityFlatteningXeSSPass(graphBuilder, sceneDepth, sceneVelocity, view, arg5, arg6, arg7, arg8);
    t_flatteningView = nullptr;

    return result;
}

void* __fastcall hkFlattenVelocity(void* graphBuilder, void* sceneDepth, void* sceneVelocity, void* view,
                                   const wchar_t* name, void* event, bool upsample)
{
    if (name == _xessVelocityName)
    {
        RecordSceneDepth(sceneDepth);
        upsample = false;
    }

    return o_FlattenVelocity(graphBuilder, sceneDepth, sceneVelocity, view, name, event, upsample);
}

// Calls visit(address, nextAddress) for each instruction until it returns false or end is reached
template <typename Visitor> void WalkInstructions(uintptr_t start, uintptr_t end, int maxCount, Visitor&& visit)
{
    auto address = start;

    for (int i = 0; i < maxCount && address < end; i++)
    {
        auto next = (uintptr_t) DetourCopyInstruction(nullptr, nullptr, (PVOID) address, nullptr, nullptr);

        if (next <= address || !visit(address, next))
            return;

        address = next;
    }
}

// Target of call rel32 or call [rip + disp32], the latter being how modular builds call other modules' exports
uintptr_t CallTarget(uintptr_t address, uintptr_t next)
{
    auto call = XeSSUnrealDecode::DecodeCall((const uint8_t*) address, next);

    if (call.kind != XeSSUnrealDecode::Call::Kind::Indirect)
        return call.target;

    uintptr_t target = 0;
    SafeRead(call.target, target);
    return target;
}

// Every plugin version calls GetSecondaryViewRectSize(View) in the flattening, View being the 4th argument, and the
// first View field it loads is ViewRect: in the flattening itself (UE4 1.x, UE5), or in the helper the XeSS wrapper
// calls right after naming the texture (UE4 2.x, modular UE5 builds)
// First call rel32 after the texture name is loaded. nameArgument tells whether the name goes in as the 5th argument
// and a true bool as the 7th, the shape of the flattening shared with XeFG:
// lea reg, [Name]; mov [rsp + 20h], reg; mov byte ptr [rsp + 30h], 1; call
uintptr_t CallAfterName(uintptr_t nameReference, bool& nameArgument)
{
    uintptr_t target = 0;
    bool fifth = false;
    bool seventh = false;

    WalkInstructions(nameReference, nameReference + 0x40, 16,
                     [&](uintptr_t address, uintptr_t next)
                     {
                         auto code = (const uint8_t*) address;

                         fifth |= XeSSUnrealDecode::StoresFifthArgument(code);
                         seventh |= XeSSUnrealDecode::SetsSeventhArgumentTrue(code);

                         if (code[0] == 0xE8)
                             target = CallTarget(address, next);

                         return target == 0;
                     });

    nameArgument = fifth && seventh;

    return target;
}

void FindViewSizeAccess(uintptr_t function, uintptr_t nameReference, uintptr_t& getSecondaryViewRectSize,
                        int32_t& viewRectOffset)
{
    using XeSSUnrealDecode::R9;
    using XeSSUnrealDecode::Rcx;

    auto walk = [&](uintptr_t start, uintptr_t end, int maxCount)
    {
        uint32_t viewRegisters = R9;

        WalkInstructions(start, end, maxCount,
                         [&](uintptr_t address, uintptr_t next)
                         {
                             if (auto target = CallTarget(address, next); target != 0)
                             {
                                 if (getSecondaryViewRectSize == 0 && (viewRegisters & Rcx))
                                     getSecondaryViewRectSize = target;

                                 viewRegisters &= ~XeSSUnrealDecode::VolatileRegisters;
                                 return true;
                             }

                             XeSSUnrealDecode::TrackViewRegisters((const uint8_t*) address, viewRegisters,
                                                                  viewRectOffset);
                             return true;
                         });
    };

    walk(function, nameReference, 64);

    if (viewRectOffset != 0)
        return;

    bool nameArgument = false;

    if (auto helper = CallAfterName(nameReference, nameArgument); helper != 0)
        walk(helper, helper + 0x400, 128);
}

// AddPasses(FRDGBuilder&, const FSceneView&, const FInputs&) const returns its FOutputs through the pointer in rdx,
// the flattening returns an FRDGTextureRef and only reads its scene depth through rdx. A store through rdx before the
// first call tells them apart.
bool StoresThroughRdx(uintptr_t function)
{
    bool stores = false;

    WalkInstructions(function, function + 0x200, 48,
                     [&](uintptr_t address, uintptr_t)
                     {
                         auto use = XeSSUnrealDecode::ClassifyRdx((const uint8_t*) address);
                         stores = use == XeSSUnrealDecode::RdxUse::Store;
                         return use == XeSSUnrealDecode::RdxUse::None;
                     });

    return stores;
}

bool HookModule(HMODULE module)
{
    // Flattened velocity texture names: UE4 plugin 1.x, UE4 plugin 2.x's XeSS wrapper, UE5 plugin's XeSS wrapper
    uintptr_t name = 0;
    for (auto textureName : { L"Upscaled Velocity Texture", L"XeSS::FlattenedVelocity", L"XeSS::UpscaledVelocity" })
    {
        name = scanner::FindWideString(module, textureName);
        if (name != 0)
            break;
    }

    if (name == 0)
        return false;

    auto reference = scanner::FindRipReference(module, name);
    if (reference == 0)
    {
        LOG_WARN("Found the XeSS plugin's velocity texture name but no code referencing it");
        return false;
    }

    auto function = scanner::GetFunctionStart(reference);
    if (function == 0)
    {
        LOG_WARN("No function entry for the XeSS plugin's velocity flattening at {:X}", reference);
        return false;
    }

    // The XeSS wrapper inlined into AddPasses
    if (StoresThroughRdx(function))
    {
        bool nameArgument = false;
        auto flattening = CallAfterName(reference, nameArgument);

        if (flattening == 0 || !nameArgument)
        {
            LOG_WARN("The XeSS plugin's velocity texture is named in {:X}, which isn't a known velocity flattening",
                     function);
            return false;
        }

        o_FlattenVelocity = (PFN_FlattenVelocity) flattening;
        _xessVelocityName = (const wchar_t*) name;

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourAttach(&(PVOID&) o_FlattenVelocity, hkFlattenVelocity);

        // Hooked() must stay false, or the input forces reversed depth and asks for a depth that never comes
        if (auto result = DetourTransactionCommit(); result != NO_ERROR)
        {
            LOG_ERROR("Hooking the UE XeSS plugin's velocity flattening at {:X} failed: {}", flattening, result);
            o_FlattenVelocity = nullptr;
            _xessVelocityName = nullptr;
            return false;
        }

        LOG_INFO("Hooked the UE XeSS plugin's velocity flattening at {:X}, called from {:X}, for scene depth and "
                 "render resolution motion vectors",
                 flattening, function);

        return true;
    }

    uintptr_t getSecondaryViewRectSize = 0;
    int32_t viewRectOffset = 0;
    FindViewSizeAccess(function, reference, getSecondaryViewRectSize, viewRectOffset);

    o_AddVelocityFlatteningXeSSPass = (PFN_AddVelocityFlatteningXeSSPass) function;

    if (getSecondaryViewRectSize != 0 && viewRectOffset > 0)
    {
        o_GetSecondaryViewRectSize = (PFN_GetSecondaryViewRectSize) getSecondaryViewRectSize;
        _viewRectOffset = viewRectOffset;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&) o_AddVelocityFlatteningXeSSPass, hkAddVelocityFlatteningXeSSPass);

    if (o_GetSecondaryViewRectSize != nullptr)
        DetourAttach(&(PVOID&) o_GetSecondaryViewRectSize, hkGetSecondaryViewRectSize);

    if (auto result = DetourTransactionCommit(); result != NO_ERROR)
    {
        LOG_ERROR("Hooking the UE XeSS plugin's velocity flattening at {:X} failed: {}", function, result);
        o_AddVelocityFlatteningXeSSPass = nullptr;
        o_GetSecondaryViewRectSize = nullptr;
        _viewRectOffset = 0;
        return false;
    }

    LOG_INFO("Hooked the UE XeSS plugin's velocity flattening at {:X} for scene depth", function);

    if (o_GetSecondaryViewRectSize != nullptr)
        LOG_INFO(
            "Hooked GetSecondaryViewRectSize at {:X}, View.ViewRect at +{:X}, for render resolution motion vectors",
            getSecondaryViewRectSize, viewRectOffset);
    else
        LOG_WARN("GetSecondaryViewRectSize: {:X}, View.ViewRect: +{:X}, motion vectors stay at display resolution",
                 getSecondaryViewRectSize, viewRectOffset);

    return true;
}
} // namespace

// Monolithic builds have the plugin in the exe. Modular builds load it as separate modules
// (<Project>-XeSSPrePass-Win64-Shipping.dll), only present once the engine loads its plugins.
void XeSSUnreal::Hook()
{
    static std::mutex mutex;
    static std::unordered_set<HMODULE> checkedModules;

    std::scoped_lock lock(mutex);

    if (Hooked())
        return;

    if (!Config::Instance()->XeSSUnrealDepth.value_or_default())
    {
        static bool logged = false;

        if (!logged)
        {
            logged = true;
            LOG_INFO("[XeSS] UnrealDepth is off, the UE XeSS plugin's depth and motion vectors are left as they are");
        }

        return;
    }

    auto exe = GetModuleHandle(nullptr);

    if (checkedModules.insert(exe).second && HookModule(exe))
        return;

    std::vector<HMODULE> modules(1024);
    DWORD needed = 0;

    if (!EnumProcessModules(GetCurrentProcess(), modules.data(), (DWORD) (modules.size() * sizeof(HMODULE)), &needed))
        return;

    modules.resize(std::min<size_t>(modules.size(), needed / sizeof(HMODULE)));

    for (auto module : modules)
    {
        if (checkedModules.contains(module))
            continue;

        wchar_t path[MAX_PATH] {};
        if (GetModuleFileNameW(module, path, MAX_PATH) == 0)
            continue;

        std::wstring fileName = std::filesystem::path(path).filename().wstring();
        std::transform(fileName.begin(), fileName.end(), fileName.begin(), ::towlower);

        // libxess*.dll is the SDK, not the plugin
        if (fileName.find(L"xess") == std::wstring::npos || fileName.starts_with(L"libxess"))
            continue;

        checkedModules.insert(module);

        if (HookModule(module))
            return;
    }
}

bool XeSSUnreal::Hooked() { return o_AddVelocityFlatteningXeSSPass != nullptr || o_FlattenVelocity != nullptr; }

ID3D12Resource* XeSSUnreal::SceneDepth(ID3D12GraphicsCommandList* commandList,
                                       const xess_d3d12_execute_params_t* params, D3D12_RESOURCE_STATES& state)
{
    if (params->pColorTexture == nullptr || _sceneDepthRdg.load() == nullptr)
        return nullptr;

    _resourceVtable = *(uintptr_t*) params->pColorTexture;
    _minWidth = params->inputColorBase.x + params->inputWidth;
    _minHeight = params->inputColorBase.y + params->inputHeight;

    ProbeSceneDepth();

    state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    return RequestCopy(commandList, params->pColorTexture);
}
