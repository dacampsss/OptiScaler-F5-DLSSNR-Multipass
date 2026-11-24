# nvngx.dll_dlssnr.dll

Neural Rendering's snippet resolves the module that owns its caller's return address and refuses
anything whose path does not contain `nvngx.dll` -- the driver core being `_nvngx.dll`. It returns
`FAIL_PlatformError` before it inspects a single argument. OptiScaler installs as `dxgi.dll` or
`winmm.dll`, so it fails that test like anything else would.

This library exists to be named correctly. It does nothing else: it forwards create, evaluate and
release, so the calls into the snippet originate from a module the snippet accepts.

The prebuilt DLL is committed because it is 12 KB, changes almost never, and has to be in the package
for a drop-in build to work at all. To rebuild it:

    cmake -S . -B build -A x64
    cmake --build build --config Release

One detail is load-bearing and not obvious. The forwarder must not `return snippetFn(...)` -- that is a
tail call, the compiler emits a `jmp`, this module's frame disappears, and the snippet then resolves its
caller to whoever called the forwarder. The result goes into a `volatile` local first.

## Model weights

It also serves the Neural Rendering "Model weights" sets. The model DLL reads its weights as resources
(`WEIGHTS_*`) when a feature is created; the forwarder patches only that DLL's import table and answers
from `<folder>\<RESOURCE>.<CRC32 of the stock resource>.bin`. A file is used only when its CRC and size match
the DLL it is read by, otherwise the DLL's own weights are used and a line is logged.

Exports:

- `dlssnr_weights_select2(dirA, dirB)`: `WEIGHTS_HT` or `WEIGHTS_PL` (some modified DLSS-NR DLLs) = Model A, Preset A, is served from `dirA`,
  `WEIGHTS_CC_SMALL_FP8*` (Model B, Preset B / Default) from `dirB`. Null or empty = the DLL's own.
  The resource name is matched exactly. Any other `WEIGHTS_*` resource now always runs the DLL's own weights,
  even with a set chosen (one log line); before, one folder was served for every `WEIGHTS_*` resource.
- `dlssnr_weights_select(dir)`: the first version, one folder for both models (kept for older OptiScaler builds).
- `dlssnr_weights_take_log(buffer, size)`: the log lines since the last call.
