// NVAPI CUDA-kernel wrapper for the DLSS-NR model: every launch goes through RealLaunch, which feeds the kernel
// profiler (ini KernelProfile); ViT reuse (Reuse bottleneck) drops or copies the bottleneck run. CPU-built.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include "nvapi.h"
#include "DlssNrNative.h"
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <array>
#include <tuple>
#include <cstring>
#include <thread>
#include <atomic>
#include "DlssNrVitReuse.h"
#include "DlssNrVitCopyKernel.h"
#include "DlssNrKernelProfile.h"
namespace DlssNrNative {namespace {
namespace fs = std::filesystem;
using Microsoft::WRL::ComPtr;
void Check(HRESULT r)
{
    if (FAILED(r))
        throw std::runtime_error("D3D12 status " + std::to_string((unsigned) r));
}
void NvCheck(NvAPI_Status r)
{
    if (r != NVAPI_OK)
        throw std::runtime_error("NVAPI status " + std::to_string(r));
}
struct Buffer{ComPtr<ID3D12Resource>gpu;uint64_t bytes=0;UINT64 address()const{return gpu->GetGPUVirtualAddress();}};
struct State
{
    std::recursive_mutex mutex;
    DlssNrKernelProfile::Profiler prof;
    std::map<NVDX_ObjectHandle, DlssNrKernelProfile::Info>
        fnInfo; // kernel census and per-group GPU timing (ini KernelProfile)
    DlssNrVitReuse::Filter vit;
    std::map<NVDX_ObjectHandle, std::pair<DlssNrVitReuse::Role, DlssNrVitReuse::KernelSet>>
        vitRole; // ViT reuse: launches of the ViT run are dropped on reused frames; each kernel set has its own rate
    // ViT reuse copy mode (plain set): our copy of the run's 2-D output per device and feature, and the copy kernel per
    // device (the device is held, so its address is not reused; failed = its load failed, not retried). Replaced
    // buffers are retired, never freed.
    struct VitCopy
    {
        Buffer buf;
        uint64_t dst = 0, bytes = 0;
    };
    std::map<std::pair<ID3D12Device*, const void*>, VitCopy> vitCopies;
    struct VitCopyKernel
    {
        ComPtr<ID3D12Device> device;
        NVDX_ObjectHandle fn = nullptr;
        bool failed = false;
    };
    std::map<ID3D12Device*, VitCopyKernel> vitCopyFn;
    std::vector<Buffer> vitRetired;
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
// Deliberately retain resources until process exit, never free under recorded GPU work.
State&S(){static State*s=new State;return*s;}
// Every real launch goes through here so the kernel profiler sees it; the caller holds S().mutex.
NvAPI_Status RealLaunch(State&s,ID3D12GraphicsCommandList*c,const NVAPI_CU_KERNEL_LAUNCH_PARAMS*k,NvU32 count){
 auto rc=s.launch(c,k,count);
 if(rc==NVAPI_OK&&k&&count&&s.prof.Recording(c)){std::vector<DlssNrKernelProfile::Info>infos;infos.reserve(count);for(NvU32 i=0;i<count;++i){auto it=s.fnInfo.find(k[i].hFunction);infos.push_back(it!=s.fnInfo.end()?it->second:DlssNrKernelProfile::Classify(""));}s.prof.Launched(c,infos);}
 return rc;}
void Barrier(ID3D12GraphicsCommandList* c)
{
    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    c->ResourceBarrier(1, &b);
}
ComPtr<ID3D12Resource>Resource(ID3D12Device*d,uint64_t bytes,D3D12_HEAP_TYPE type,D3D12_RESOURCE_STATES state){D3D12_RESOURCE_DESC r{};r.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;r.Width=bytes;r.Height=1;r.DepthOrArraySize=1;r.MipLevels=1;r.SampleDesc.Count=1;r.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;if(type==D3D12_HEAP_TYPE_DEFAULT)r.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;D3D12_HEAP_PROPERTIES h{};h.Type=type;ComPtr<ID3D12Resource>out;Check(d->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&r,state,nullptr,IID_PPV_ARGS(&out)));return out;}
Buffer Allocate(ID3D12Device* d, uint64_t bytes)
{
    return { Resource(d, std::max<uint64_t>(bytes, 256), D3D12_HEAP_TYPE_DEFAULT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
             std::max<uint64_t>(bytes, 256) };
}
void LaunchBlob(ID3D12GraphicsCommandList*c,NVDX_ObjectHandle f,const void*p,unsigned size,NVAPI_DIM3 grid,NVAPI_DIM3 block,unsigned shared=0){NVAPI_CU_KERNEL_LAUNCH_PARAMS k{};k.hFunction=f;k.gridDim=grid;k.blockDim=block;k.pParams=p;k.paramSize=size;k.dynSharedMemBytes=shared;NvCheck(RealLaunch(S(),c,&k,1));}
NvAPI_Status __cdecl CreateFunction(ID3D12Device* d, NVDX_ObjectHandle m, const char* n, NVDX_ObjectHandle* out)
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> apiGuard(s.mutex);
    auto rc = s.createFunction(d, m, n, out);
    if (rc == 0 && n)
    {
        {
            std::lock_guard<std::recursive_mutex> g(s.mutex);
            s.fnInfo[*out] = DlssNrKernelProfile::Classify(n);
        }
        {
            auto role = DlssNrVitReuse::RoleOf(n);
            if (role != DlssNrVitReuse::Role::None)
            {
                std::lock_guard<std::recursive_mutex> g(s.mutex);
                s.vitRole[*out] = { role, DlssNrVitReuse::SetOf(n) };
            }
        }
    }
    return rc;
}
// ViT reuse: true when this call needs more than a plain launch (a launch dropped, a barrier or a copy to record); `kept` then holds the launches that go out.
// Only launches recorded inside an NR evaluation are looked at. `gap`, `copy`, `copyAt`, `copyFrom`: see Filter::Split. A run's first launch is identified by
// its parameters and geometry (FNV-1a), so a moved or resized network is computed again in copy mode.
bool VitDrop(State&s,const NVAPI_CU_KERNEL_LAUNCH_PARAMS*k,NvU32 count,std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS>&kept,ptrdiff_t&gap,DlssNrVitReuse::Copy&copy,ptrdiff_t&copyAt,ptrdiff_t&copyFrom){
 if(!k||!count||!s.vit.Evaluating())return false;
 const auto sigOf=[](const NVAPI_CU_KERNEL_LAUNCH_PARAMS&l){uint64_t h=1469598103934665603ull;const auto mix=[&](const void*p,size_t n){for(size_t i=0;i<n;++i){h^=((const unsigned char*)p)[i];h*=1099511628211ull;}};if(l.pParams)mix(l.pParams,l.paramSize);mix(&l.gridDim,sizeof l.gridDim);mix(&l.blockDim,sizeof l.blockDim);return h;};
 return s.vit.Split(k,count,[&](const NVAPI_CU_KERNEL_LAUNCH_PARAMS&l){auto it=s.vitRole.find(l.hFunction);return it==s.vitRole.end()?std::pair{DlssNrVitReuse::Role::None,DlssNrVitReuse::KernelSet::Unknown}:it->second;},sigOf,kept,gap,copy,copyAt,copyFrom);}
// ViT reuse copy mode: record the save (the run's 2-D output into our buffer) or the restore (back into it). `end` is the run's last launch;
// DlssNrVitReuse::CheckOutput says where its 2-D output is and how big. Returns why it could not, or nullptr.
const char* VitCopyRecord(State&s,ID3D12GraphicsCommandList*c,DlssNrVitReuse::Copy kind,const NVAPI_CU_KERNEL_LAUNCH_PARAMS&end){
 struct{uint64_t dst;}e{};uint64_t bytes=0;const uint64_t threads=(uint64_t)end.gridDim.x*end.gridDim.y*end.gridDim.z*end.blockDim.x*end.blockDim.y*end.blockDim.z;
 if(const char*why=DlssNrVitReuse::CheckOutput(end.pParams,end.paramSize,threads,s.vit.LastSet(),e.dst,bytes))return why;
 try{ComPtr<ID3D12Device>dev;Check(c->GetDevice(IID_PPV_ARGS(&dev)));auto&k=s.vitCopyFn[dev.Get()];if(k.failed)return "the copy kernel could not be loaded";
  if(!k.fn){k.device=dev;NVDX_ObjectHandle m=nullptr;if(!s.createModule||!s.createFunction||s.createModule(dev.Get(),DlssNrVitCopyKernel::fatbin,(NvU32)sizeof DlssNrVitCopyKernel::fatbin,&m)!=NVAPI_OK||s.createFunction(dev.Get(),m,"nr_vit_copy16",&k.fn)!=NVAPI_OK){if(m&&s.destroyModule)s.destroyModule(dev.Get(),m);k.fn=nullptr;k.failed=true;return "the copy kernel could not be loaded";}s.fnInfo[k.fn]=DlssNrKernelProfile::Classify("nr_vit_copy16");}
  const NVDX_ObjectHandle fn=k.fn;auto&v=s.vitCopies[{dev.Get(),s.vit.Feature()}];
  if(kind==DlssNrVitReuse::Copy::Save){if(v.buf.bytes<bytes){if(v.buf.gpu)s.vitRetired.push_back(v.buf);v.buf=Allocate(dev.Get(),bytes);}v.dst=e.dst;v.bytes=bytes;}
  else if(!v.buf.gpu||v.dst!=e.dst||v.bytes!=bytes)return "the saved ViT output does not match the network";
  const uint64_t n=bytes/16;struct Params{uint64_t dst,src,n;}p{kind==DlssNrVitReuse::Copy::Save?v.buf.address():e.dst,kind==DlssNrVitReuse::Copy::Save?e.dst:v.buf.address(),n};
  LaunchBlob(c,fn,&p,sizeof p,{(unsigned)std::min<uint64_t>((n+255)/256,1024),1,1},{256,1,1});}
 catch(const std::exception&){return "the copy could not be recorded";}
 return nullptr;}
NvAPI_Status __cdecl Launch(ID3D12GraphicsCommandList*c,const NVAPI_CU_KERNEL_LAUNCH_PARAMS*k,NvU32 count){auto&s=S();std::lock_guard<std::recursive_mutex>g(s.mutex);
 // A UAV barrier where the skipped run would have been: without it the kernel before the gap and the one after it can overlap on the
 // GPU. The run's own kernels kept them apart. On the plain set the kernel after it is the run's kept last kernel (see DlssNrVitReuse.h).
 {std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS>kept;ptrdiff_t gap=-1,copyAt=-1,copyFrom=-1;DlssNrVitReuse::Copy copy=DlssNrVitReuse::Copy::None;if(VitDrop(s,k,count,kept,gap,copy,copyAt,copyFrom)){
  // Copy mode: the copy goes at copyAt with a UAV barrier on each side, so it sees the 2-D output complete (save) or is complete before the
  // decoder reads it (restore, which also stands where the skipped run was). If it cannot be recorded, Reuse goes off; a restore that failed
  // falls back to the run's last kernel, as without copy mode: right where the DLL keeps the 1-D result, one stale frame where it does not.
  if(copy!=DlssNrVitReuse::Copy::None){if(copyAt>0){auto r=RealLaunch(s,c,kept.data(),(NvU32)copyAt);if(r!=NVAPI_OK)return r;}Barrier(c);
   if(const char*why=VitCopyRecord(s,c,copy,k[copyFrom])){s.vit.Disable(why);if(copy==DlssNrVitReuse::Copy::Restore){auto r=RealLaunch(s,c,&k[copyFrom],1);if(r!=NVAPI_OK)return r;}}
   Barrier(c);return (size_t)copyAt<kept.size()?RealLaunch(s,c,kept.data()+copyAt,(NvU32)(kept.size()-copyAt)):NVAPI_OK;}
  if(gap<0)return kept.empty()?NVAPI_OK:RealLaunch(s,c,kept.data(),(NvU32)kept.size());
  // the kept launches of this call on either side of the gap go out as two chains with the barrier between them
  if(gap>0){auto r=RealLaunch(s,c,kept.data(),(NvU32)gap);if(r!=NVAPI_OK)return r;}Barrier(c);return (size_t)gap<kept.size()?RealLaunch(s,c,kept.data()+gap,(NvU32)(kept.size()-gap)):NVAPI_OK;}}
 return RealLaunch(s, c, k, count);
}
NvAPI_Status __cdecl DestroyFunction(ID3D12Device* d, NVDX_ObjectHandle f)
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    s.vitRole.erase(f);
    s.fnInfo.erase(f);
    return s.destroyFunction(d, f);
}
NvAPI_Status __cdecl DestroyModule(ID3D12Device* d, NVDX_ObjectHandle m)
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    s.vit.Clear();
    return s.destroyModule(d, m);
}
} // namespace
void* WrapNvapi(unsigned id, void* raw)
{
    if (!raw)
        return raw;
    auto& s = S();
    std::lock_guard<std::recursive_mutex> apiGuard(s.mutex);
    switch (id)
    {
    case 0xad1a677d: // CreateCuModule: not wrapped, only kept to load the ViT copy kernel
        s.createModule = (decltype(s.createModule)) raw;
        return raw;
    case 0xe2436e22:
        s.createFunction = (decltype(s.createFunction)) raw;
        return (void*) &CreateFunction;
    case 0x24973538:
        s.launch = (decltype(s.launch)) raw;
        return (void*) &Launch;
    case 0x41c65285:
        s.destroyModule = (decltype(s.destroyModule)) raw;
        return (void*) &DestroyModule;
    case 0xdf295ea6:
        s.destroyFunction = (decltype(s.destroyFunction)) raw;
        return (void*) &DestroyFunction;
    default:
        return raw;
    }
}
void BeginEvaluate(const void* feature, bool reset, unsigned every, unsigned everyPlain, long long slot,
                   ID3D12GraphicsCommandList* cmd, bool profile, unsigned copySets)
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    s.vit.Begin(feature, reset, every, slot, everyPlain, copySets);
    s.prof.Begin(cmd, profile);
}
bool EndEvaluate(ID3D12GraphicsCommandList* cmd)
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    s.prof.End(cmd);
    const bool wasOff = s.vit.Disabled();
    const bool wellFormed = s.vit.End();
    return (!wellFormed || s.vit.Disabled()) && !wasOff;
}
std::vector<std::string> TakeProfileReports()
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    return s.prof.TakeReports();
}
std::string VitStatus()
{
    auto& s = S();
    std::lock_guard<std::recursive_mutex> g(s.mutex);
    return s.vit.Status();
}
bool VitPlainKernels() { return S().vit.LastSet() == DlssNrVitReuse::KernelSet::Plain; }
bool VitKernelsSeen() { return S().vit.LastSet() != DlssNrVitReuse::KernelSet::Unknown; }
// Not in DlssNrNative.h: only DlssNrNative_Vk.cpp drives the filter directly (one launch per call there).
DlssNrVitReuse::Filter& VitFilter() { return S().vit; }
const char* VitKernelSet()
{
    switch (S().vit.LastSet())
    {
    case DlssNrVitReuse::KernelSet::Fp8:
        return "FP8";
    case DlssNrVitReuse::KernelSet::Plain:
        return "plain FP16";
    default:
        return "not seen yet";
    }
}
} // namespace DlssNrNative
