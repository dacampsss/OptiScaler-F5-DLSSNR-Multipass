// Vulkan builds of the frozen v59 shader (precompile/dlssnr.hlsl). Under VK_MODE that file leaves out its two auxiliary
// textures (gAux, gAux2: the Direct NR modes, not used by any Superior pass the Vulkan build has) but a few of its modes
// still name them, so the file does not compile as it stands. They are declared here, before the file is included, on
// bindings the Superior pass layout does not carry: no pass built from this include reads them, and the compiler drops
// the unused declarations. The frozen file itself stays byte for byte as it is (its hash is pinned).
[[vk::binding(9, 0)]]
Texture2D<float4> gAux : register(t4);
[[vk::binding(10, 0)]]
Texture2D<float4> gAux2 : register(t5);
