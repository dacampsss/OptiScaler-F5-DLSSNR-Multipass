# Rebuilds the Superior Scaling SPIR-V blobs for native Vulkan (OptiScaler/shaders/dlssnr/superior/precompile_vk/*_Shader_Vk.h)
# with the repository's own dxc (OptiScaler/shaders/shader_tools/dxc.exe). PowerShell only, no Python.
#
# The Vulkan Superior pass (shaders/dlssnr/DlssNrSuperior_Vk.cpp) runs the No DLSS path of the D3D12 one, from the same
# sources, built with VK_MODE:
#   frozen v59 shader      superior/dlssnr_vk.hlsl (VkAux.hlsli + precompile/dlssnr.hlsl) -> dlssnr_superior_area
#                          superior/RawLow_vk.hlsl (VkAux.hlsli + RawLow.hlsl)           -> host_low_raw
#   guide passes           superior/SpatialGuides.hlsl, JointGuides.hlsl                 -> host_uniform_*guides*
#   live NR shader         precompile/dlssnr.hlsl + superior/HostResolve.hlsli           -> host_resolve_guided_raw
#                          (and the signed-residual resolves of the private DLSS: host_resolve_hdr/sdr, host_normal_*)
#   private DLSS           superior/PrivateCarrier_vk.hlsl (encode, decode), PrivateMotion.hlsl, the ghost field shaders
#                          (CompactGhost, GhostContrastHistory, CoarseGhostDilate, GhostBiasMask, GhostOutwardFeather) -> private_*
#   frozen v59 shader, more modes: the generic pass (Mode 7, the private DLSS's unit exposure), the carrier decode
#                          (Mode 38) and the signed residual's low field (Mode 18)   -> dlssnr_superior_*
# Every entry point is renamed CSMain in the module, so the host makes all pipelines the same way.
#
# The D3D12 blobs are not touched: the sources only gained code under VK_MODE, and tools/compile_nr_interpass.ps1 -Check
# still has to say that every D3D12 blob matches. A change to any source here must be followed by this script and the result
# committed. -Check builds into a temporary copy and compares instead of writing (exit 1 on any difference).
param([switch]$Check)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$shaders = Join-Path $root "OptiScaler/shaders/dlssnr"
$dxc = Join-Path $root "OptiScaler/shaders/shader_tools/dxc.exe"
$out = Join-Path $shaders "superior/precompile_vk"
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("superior-vk-blobs-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $scratch | Out-Null
New-Item -ItemType Directory -Force $out | Out-Null
$built = New-Object System.Collections.Generic.List[string]

# Byte-for-byte what shader_tools/create_header.py writes, with trailing spaces trimmed and CRLF lines.
function Write-Header([string]$spv, [string]$header, [string]$name) {
    $bytes = [IO.File]::ReadAllBytes($spv)
    $sb = New-Object Text.StringBuilder
    [void]$sb.Append("#pragma once`n`ninline static const unsigned char $name[] = {`n    ")
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        [void]$sb.Append(('0x{0:x2}' -f $bytes[$i]))
        if ($i -lt $bytes.Length - 1) { [void]$sb.Append(', ') }
        if ((($i + 1) % 12) -eq 0) { [void]$sb.Append("`n    ") }
    }
    [void]$sb.Append("`n};`n")
    $lines = $sb.ToString().TrimEnd("`n").Split("`n") | ForEach-Object { $_.TrimEnd() }
    [IO.File]::WriteAllText($header, (($lines -join "`r`n") + "`r`n"), [Text.UTF8Encoding]::new($false))
}

function Build([string]$source, [string]$entry, [string[]]$defines, [string]$stem, [string[]]$bindings = @()) {
    $spv = Join-Path $scratch "${stem}_Shader_Vk.spv"
    $dxcArgs = @('-spirv', '-T', 'cs_6_0', '-E', $entry, '-fspv-entrypoint-name=CSMain', '-O3', '-Qstrip_debug', '-D', 'VK_MODE') +
        $defines + $bindings + @((Split-Path $source -Leaf), '-Fo', $spv)
    Push-Location (Split-Path $source -Parent)
    try { & $dxc @dxcArgs | Out-Null } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "DXC failed: $stem" }
    Write-Header $spv (Join-Path $scratch "${stem}_Shader_Vk.h") "${stem}_spv"
    $built.Add($stem)
}

try {
    $superior = Join-Path $shaders "superior"
    Build (Join-Path $superior "dlssnr_vk.hlsl") "CSMain" @("-D", "DLSSNR_SUPERIOR_SCALING=1", "-D", "DLSSNR_SUPERIOR_AREA=1", "-D", "VK_TARGET_RGBA16F=1") "dlssnr_superior_area"
    Build (Join-Path $superior "RawLow_vk.hlsl") "RawLowMain" @("-D", "VK_TARGET_RGBA16F=1") "host_low_raw"

    Build (Join-Path $superior "SpatialGuides.hlsl") "SuperiorUniformSurfaceGuidesMain" @() "host_uniform_surface_guides"
    foreach ($minified in @($false, $true)) {
        $defines = @()
        if ($minified) { $defines += @("-D", "DLSSNR_JOINT_MINIFY=1") }
        $entry = if ($minified) { "SuperiorUniformJointMinifiedGuidesMain" } else { "SuperiorUniformJointGuidesMain" }
        $stem = "host_uniform_joint_" + $(if ($minified) { "minified_" } else { "" }) + "guides"
        Build (Join-Path $superior "JointGuides.hlsl") $entry $defines $stem
        Build (Join-Path $superior "JointGuides.hlsl") $entry ($defines + @("-D", "DLSSNR_QUANTIZED_DEPTH=1")) ($stem + "_quantized")
    }

    $live = Join-Path $shaders "precompile/dlssnr.hlsl"
    Build $live "CSMain" @("-D", "DLSSNR_SUPERIOR_HOST_RESOLVE=1", "-D", "DLSSNR_SUPERIOR_HOST_RAW=1", "-D", "DLSSNR_SUPERIOR_GUIDED_RAW=1") "host_resolve_guided_raw"

    # The signed-residual resolves (a private DLSS ran): the live shader without the raw-answer define, the plain ones and
    # the colour-mapping specializations of the D3D12 set that an unpacked frame can reach.
    $hostResolve = @("-D", "DLSSNR_SUPERIOR_HOST_RESOLVE=1")
    Build $live "CSMain" $hostResolve "host_resolve_hdr"
    Build $live "CSMain" ($hostResolve + @("-D", "DLSSNR_SUPERIOR_HOST_SDR=1")) "host_resolve_sdr"
    foreach ($variant in @("sdr", "hdr0", "hdr2", "hdr4", "hdr6")) {
        $pass = if ($variant -eq "sdr") { 1 } else { 0 }
        $mapping = if ($pass) { 0 } else { [int]$variant.Substring(3) }
        $defines = $hostResolve + @("-D", "DLSSNR_SUPERIOR_HOST_NORMAL=1", "-D", "DLSSNR_SUPERIOR_HOST_PASSTHROUGH=$pass",
                                    "-D", "DLSSNR_SUPERIOR_HOST_MAPPING=$mapping")
        if ($pass) { $defines += @("-D", "DLSSNR_SUPERIOR_HOST_SDR=1") }
        Build $live "CSMain" $defines "host_normal_$variant"
    }

    # More modes of the frozen v59 shader (through the same wrapper as the Area downsample).
    Build (Join-Path $superior "dlssnr_vk.hlsl") "CSMain" @("-D", "VK_TARGET_R32F=1") "dlssnr_superior_generic"
    Build (Join-Path $superior "dlssnr_vk.hlsl") "SuperiorCarrierMain" @("-D", "DLSSNR_SUPERIOR_SCALING=1", "-D", "VK_TARGET_RGBA16F=1") "dlssnr_superior_carrier"
    Build (Join-Path $superior "dlssnr_vk.hlsl") "CSMain" @("-D", "DLSSNR_SUPERIOR_SCALING=1", "-D", "DLSSNR_SUPERIOR_LOW=1", "-D", "VK_TARGET_RGBA16F=1") "dlssnr_superior_low"

    # The private DLSS's own shaders carry no vk::binding (except through the frozen shader they include): the host's
    # descriptor layout is fixed here, register by register. b0 the constants, t0-t3 bindings 1-4, t5 binding 8, t4 binding 9
    # (the frozen shader's gAux), u0-u1 bindings 5-6, s0 binding 7 (DlssNrSuperior_Vk.h).
    $layout = @("-fvk-bind-register", "b0", "0", "0", "0", "-fvk-bind-register", "t0", "0", "1", "0",
                "-fvk-bind-register", "t1", "0", "2", "0", "-fvk-bind-register", "t2", "0", "3", "0",
                "-fvk-bind-register", "t3", "0", "4", "0", "-fvk-bind-register", "t4", "0", "9", "0",
                "-fvk-bind-register", "t5", "0", "8", "0", "-fvk-bind-register", "u0", "0", "5", "0",
                "-fvk-bind-register", "u1", "0", "6", "0", "-fvk-bind-register", "s0", "0", "7", "0")
    $carrier = Join-Path $superior "PrivateCarrier_vk.hlsl"
    Build $carrier "PrivateEncode" @() "private_encode" $layout
    Build $carrier "PrivateDecode" @() "private_fade" $layout
    Build $carrier "PrivateDecode" @("-D", "SUPERIOR_GHOST_DECODE=1", "-D", "SUPERIOR_GHOST_DECODE_SEED=1") "private_ghost_decode_seed" $layout
    Build (Join-Path $superior "PrivateMotion.hlsl") "SampleMotion" @() "private_cnn_motion" $layout
    Build (Join-Path $superior "CompactGhost.hlsl") "ClearVisibility" @() "private_ghost_clear" $layout
    foreach ($stage in @(1, 2)) {
        Build (Join-Path $superior "CompactGhost.hlsl") "CompactGhostMain" @("-D", "DLSSNR_COMPACT_GHOST_STAGE=$stage", "-D", "DLSSNR_GHOST_STEP=4") "private_ghost_coarse_$stage" $layout
    }
    Build (Join-Path $superior "GhostContrastHistory.hlsl") "GhostContrastHistory" @("-D", "DLSSNR_CONTRAST_COMPACT=1", "-D", "DLSSNR_GHOST_STEP=4") "private_ghost_reference" $layout
    Build (Join-Path $superior "CoarseGhostDilate.hlsl") "CoarseGhostDilate" @("-D", "DLSSNR_COARSE_RADIUS=1", "-D", "DLSSNR_GHOST_CONTRAST=1") "private_ghost_dilate" $layout
    Build (Join-Path $superior "GhostBiasMask.hlsl") "GhostBiasMask" @() "private_ghost_bias" $layout
    Build (Join-Path $superior "GhostOutwardFeather.hlsl") "GhostOutwardFeather" @("-D", "DLSSNR_GHOST_FEATHER_STAGE=1", "-D", "DLSSNR_GHOST_FEATHER_INPLACE=1") "private_ghost_rings_inplace" $layout

    # Every shipped blob must come from this script, and nothing else may sit beside them.
    $shipped = @(Get-ChildItem $out -Filter "*_Shader_Vk.h" | ForEach-Object { $_.Name -replace '_Shader_Vk\.h$', '' })
    $orphans = @($shipped | Where-Object { $built -notcontains $_ })
    $different = New-Object System.Collections.Generic.List[string]
    foreach ($stem in $built) {
        foreach ($ext in @("_Shader_Vk.h")) {
            $new = Join-Path $scratch "$stem$ext"
            $old = Join-Path $out "$stem$ext"
            if ($Check) {
                if (-not ((Test-Path $old) -and (Get-FileHash $new).Hash -eq (Get-FileHash $old).Hash)) { $different.Add("$stem$ext") }
            } else {
                Copy-Item $new $old -Force
            }
        }
        Write-Host "$stem SHA256=$((Get-FileHash (Join-Path $scratch "${stem}_Shader_Vk.spv")).Hash)"
    }
    Write-Host "$($built.Count) blobs built with $dxc"
    if ($orphans.Count) { Write-Host "Not built by this script: $($orphans -join ', ')" }
    if ($Check -and $different.Count) { Write-Host "Differs from the committed blob: $($different -join ', ')" }
    if ($Check -and ($orphans.Count -or $different.Count)) { exit 1 }
    if ($Check) { Write-Host "All $($built.Count) blobs and headers match byte for byte" }
} finally {
    Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue
}
