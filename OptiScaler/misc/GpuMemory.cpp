#include "pch.h"
#include "GpuMemory.h"

#include "GpuMemTally.h"

#include <Config.h>
#include <SysUtils.h>
#include <Util.h>

#include <atomic>
#include <unordered_map>

#include <dxgi1_4.h>

namespace
{
std::atomic<bool> g_active { false };
std::atomic<bool> g_protect { false };
std::atomic<int> g_scope { 0 };
std::atomic<bool> g_haveAdapter { false };
IDXGIAdapter3* g_adapter = nullptr; // never released: lives as long as the process
gpumem::Tally g_tally;

std::mutex g_nameMutex;
std::unordered_map<HMODULE, std::string> g_names;

// {6F1D7A52-2C48-4C7B-9E0A-5B3F8E1D44A9}
const GUID kTrackerGuid = { 0x6f1d7a52, 0x2c48, 0x4c7b, { 0x9e, 0x0a, 0x5b, 0x3f, 0x8e, 0x1d, 0x44, 0xa9 } };

std::string BucketOf(HMODULE module)
{
    if (module == nullptr)
        return "other";

    {
        std::lock_guard lock(g_nameMutex);
        if (auto it = g_names.find(module); it != g_names.end())
            return it->second;
    }

    // GetModuleFileNameW takes the loader lock: never while holding g_nameMutex (a DllMain that creates a resource
    // would wait on us while we wait on it). Two threads may resolve the same module; the first insert wins.
    std::string name;
    if (module == dllModule)
        name = "OptiScaler";
    else if (module == GetModuleHandleW(nullptr))
        name = "game";
    else
    {
        wchar_t path[MAX_PATH] {};
        const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
        if (n == 0 || n >= MAX_PATH)
            name = "other";
        else
            name = gpumem::ModuleFileName(std::wstring_view(path, n));
    }

    std::lock_guard lock(g_nameMutex);
    return g_names.emplace(module, std::move(name)).first->second;
}

// The budget and usage, asked outside the tally's lock: Add() also runs inside D3D12/DXGI object destruction (the
// tracker's Release), so no DXGI call may happen while that lock is held
std::string BudgetText()
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info {};
    if (!g_haveAdapter.load() || g_adapter == nullptr ||
        FAILED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return {};
    char buf[96];
    snprintf(buf, sizeof(buf), "; budget %llu MB, usage %llu MB", (unsigned long long) (info.Budget >> 20),
             (unsigned long long) (info.CurrentUsage >> 20));
    return buf;
}

void Emit(std::string line, bool withBudget = true)
{
    if (line.empty())
        return;

    // Inside the closing bracket of the tally's line; not from the tracker's Release (D3D12/DXGI may hold locks there)
    if (const std::string budget = withBudget ? BudgetText() : std::string {}; !budget.empty() && line.back() == ')')
        line.insert(line.size() - 1, budget);

    spdlog::info("{}", line);
}

// Lives as private data of the counted resource or heap; D3D12 releases it when the object dies
class Tracker final : public IUnknown
{
  public:
    Tracker(std::string bucket, bool host, uint64_t bytes, uint64_t prot)
        : _bucket(std::move(bucket)), _host(host), _bytes(bytes), _prot(prot)
    {
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
    {
        if (riid == __uuidof(IUnknown))
        {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return ++_refs; }

    uint64_t Bytes() const { return _bytes; }

    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG n = --_refs;
        if (n == 0)
        {
            // Not at process exit: the line it could produce would be logged into a closed log
            if (!State::Existing() || !State::Existing()->isShuttingDown)
                Emit(g_tally.Add(_bucket, _host, -(int64_t) _bytes, GetTickCount64(), -(int64_t) _prot), false);
            delete this;
        }
        return n;
    }

  private:
    std::atomic<ULONG> _refs { 1 };
    std::string _bucket;
    bool _host;
    uint64_t _bytes;
    uint64_t _prot;
};
} // namespace

bool GpuMemory::Init()
{
    auto config = Config::Instance();
    const bool on = config->LogGpuMemory.value_or(config->LogToFile.value_or_default());
    const int scope = config->ProtectMemory.value_or_default();
    const bool protect = scope != gpumem::ProtectOff;
    g_active.store(on);
    g_protect.store(protect);
    g_scope.store(scope);

    return on || protect;
}

bool GpuMemory::Active() { return g_active.load(std::memory_order_relaxed); }

bool GpuMemory::ProtectOn() { return g_protect.load(std::memory_order_relaxed); }

bool GpuMemory::ShouldProtect(int heapType, int memoryPool, void* caller)
{
    if (!ProtectOn() || !gpumem::IsGpuMemory(heapType, memoryPool))
        return false;

    HMODULE module = Util::GetCallerModule(caller);
    if (module == nullptr)
        return false;

    return gpumem::ProtectedUnder(g_scope.load(), BucketOf(module));
}

void GpuMemory::NoteDevice(ID3D12Device* device)
{
    if (!Active() || device == nullptr || g_haveAdapter.load())
        return;

    static std::mutex m;
    std::lock_guard lock(m);
    if (g_haveAdapter.load())
        return;

    IDXGIFactory4* factory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory != nullptr)
    {
        IDXGIAdapter3* adapter = nullptr;
        if (SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
            g_adapter = adapter;
        factory->Release();
    }

    g_haveAdapter.store(true); // also when it failed: no retry per allocation
}

void GpuMemory::Note(IUnknown* object, uint64_t bytes, int heapType, int memoryPool, void* caller,
                     uint64_t protectedBytes)
{
    if (!Active() || object == nullptr || bytes == 0)
        return;

    ID3D12Object* d3dObject = nullptr;
    if (FAILED(object->QueryInterface(IID_PPV_ARGS(&d3dObject))) || d3dObject == nullptr)
        return;

    // Hooked entry points can call each other inside D3D12: count an object once
    UINT size = 0;
    if (d3dObject->GetPrivateData(kTrackerGuid, &size, nullptr) != DXGI_ERROR_NOT_FOUND)
    {
        d3dObject->Release();
        return;
    }

    const std::string bucket = BucketOf(Util::GetCallerModule(caller));
    const bool host = !gpumem::IsGpuMemory(heapType, memoryPool);

    auto* tracker = new Tracker(bucket, host, bytes, protectedBytes);
    const HRESULT hr = d3dObject->SetPrivateDataInterface(kTrackerGuid, tracker);
    tracker->Release(); // D3D12 holds its own reference now
    d3dObject->Release();

    if (SUCCEEDED(hr))
        Emit(g_tally.Add(bucket, host, (int64_t) bytes, GetTickCount64(), (int64_t) protectedBytes));
}

void GpuMemory::LogFinal()
{
    if (!Active())
        return;

    Emit(g_tally.Final());
}

void GpuMemory::NoteResidency(UINT numObjects, IUnknown* const* objects, const void* priorities, void* caller)
{
    if (!Active() || numObjects == 0 || objects == nullptr)
        return;

    const std::string module = BucketOf(Util::GetCallerModule(caller));
    const auto* levels = static_cast<const D3D12_RESIDENCY_PRIORITY*>(priorities);

    for (UINT i = 0; i < numObjects; ++i)
    {
        int64_t bytes = -1;
        if (objects[i] != nullptr)
        {
            ID3D12Object* obj = nullptr;
            if (SUCCEEDED(objects[i]->QueryInterface(IID_PPV_ARGS(&obj))) && obj != nullptr)
            {
                IUnknown* tracker = nullptr;
                UINT size = sizeof(tracker);
                if (SUCCEEDED(obj->GetPrivateData(kTrackerGuid, &size, &tracker)) && tracker != nullptr)
                {
                    // Our own class, attached by Note(): read what it holds, then give the reference back
                    bytes = (int64_t) static_cast<Tracker*>(tracker)->Bytes();
                    tracker->Release();
                }
                obj->Release();
            }
        }

        g_tally.NoteResidency(module, levels != nullptr ? (uint32_t) levels[i] : 0, bytes);
    }

    Emit(g_tally.Due(GetTickCount64()));
}
