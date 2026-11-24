#pragma once

#include <d3d12.h>

// GPU memory by module (misc/GpuMemTally.h holds the bookkeeping). The D3D12 device hooks report every committed
// resource and heap here; the tracker attached to each one takes it off again when D3D12 destroys it.
namespace GpuMemory
{
// Reads [Log] LogGpuMemory (auto = on with LogToFile) and [Memory] ProtectMemory (auto = fg) once and says whether the
// allocation hooks are wanted (either one). Call before hooking.
bool Init();
bool Active();    // the log line is on
bool ProtectOn(); // allocations of the modules in the scope get the high residency priority

// The caller (return address of the hook) is in the ProtectMemory scope (fg: f5low; all: OptiScaler, f5low, the NVIDIA
// runtimes) and the allocation is GPU memory: it should get D3D12_RESIDENCY_PRIORITY_HIGH. False when ProtectMemory is
// off.
bool ShouldProtect(int heapType, int memoryPool, void* caller);

// Remembers the game device's adapter for the budget and usage in the log line (once; cheap to call again)
void NoteDevice(ID3D12Device* device);

// A committed resource or heap that was just created. `object` is what the game got (any interface), `bytes` its size,
// `heapType`/`memoryPool` the D3D12 heap properties it lives in, `caller` the hook's return address.
// `protectedBytes`: how much of it the hook gave the high priority (0 if none).
void Note(IUnknown* object, uint64_t bytes, int heapType, int memoryPool, void* caller, uint64_t protectedBytes);

// A call of the game or a library to SetResidencyPriority (not ours): tallied per calling module for the log line
void NoteResidency(UINT numObjects, IUnknown* const* objects, const void* priorities, void* caller);

// The totals line now, at process exit
void LogFinal();
} // namespace GpuMemory
