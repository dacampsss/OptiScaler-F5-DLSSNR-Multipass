// Storage image formats under VK_MODE (the Vulkan builds of the private DLSS shaders). Each RWTexture2D is declared with
// the format of the image the host binds to it, so the SPIR-V is valid for exactly that view (no format inferred from the
// element type, no typed-access features beyond the core ones). D3D12 builds see empty macros: nothing changes there.
// The two-component integer state (a packed visibility state, the motion probe's count) is stored in RGBA32 UINT on
// Vulkan: R32G32 is not a storage format every device offers.
#if defined(VK_MODE)
#define VK_FORMAT(name) [[vk::image_format(name)]]
#define STATE_T uint4
#define STATE_VALUE(v) uint4((v), 0, 0)
#else
#define VK_FORMAT(name)
#define STATE_T uint2
#define STATE_VALUE(v) (v)
#endif
