#pragma once

// The few x64 instruction forms inputs/XeSS_UE_Dx12.cpp reads to find Intel's UE XeSS plugin velocity flattening, its
// View.ViewRect offset and GetSecondaryViewRectSize. Pure byte decoding of one instruction, so it is host tested
// (tests/xess_ue_decode_smoke.cpp). The caller walks instructions with DetourCopyInstruction and owns memory safety:
// every function reads at most 8 bytes from code.
//
// Ported from upstream OptiScaler 7e941561 (FakeMichau, GPL-3.0), split out of XeSS_UE_Dx12.cpp.

#include <cstdint>
#include <cstring>

namespace XeSSUnrealDecode
{
// General purpose registers as bits, rax = 0 ... r15 = 15
constexpr uint32_t Rcx = 1u << 1;
constexpr uint32_t R9 = 1u << 9;

// Lost across a call in the Microsoft x64 convention
constexpr uint32_t VolatileRegisters =
    (1u << 0) | (1u << 1) | (1u << 2) | (1u << 8) | (1u << 9) | (1u << 10) | (1u << 11);

inline int32_t ReadInt32(const uint8_t* code)
{
    int32_t value = 0;
    memcpy(&value, code, sizeof(value));
    return value;
}

struct Call
{
    enum class Kind
    {
        None,
        Direct,  // call rel32, target is the callee
        Indirect // call [rip + disp32], target is the pointer slot holding the callee (imports in modular builds)
    };

    Kind kind = Kind::None;
    uintptr_t target = 0;
};

// next is the address of the following instruction, where rip relative displacements start from
inline Call DecodeCall(const uint8_t* code, uintptr_t next)
{
    if (code[0] == 0xE8)
        return { Call::Kind::Direct, next + ReadInt32(code + 1) };

    if (code[0] == 0x48)
        code++;

    if (code[0] == 0xFF && code[1] == 0x15)
        return { Call::Kind::Indirect, next + ReadInt32(code + 2) };

    return {};
}

// Follows which registers hold the View pointer through register copies and reports the first load from
// [viewRegister + disp32] in loadOffset (left alone once set). Decodes the forms the plugin uses to copy View and read
// its fields: mov r64, r64, mov r, [r + disp32], movups xmm, [r + disp32] and lea. A 32-bit register write
// zero-extends, so it ends that register holding View.
inline void TrackViewRegisters(const uint8_t* code, uint32_t& viewRegisters, int32_t& loadOffset)
{
    uint8_t rex = 0;

    if ((code[0] & 0xF0) == 0x40)
    {
        rex = code[0];
        code++;
    }

    auto movups = code[0] == 0x0F && code[1] == 0x10;
    auto opcode = code[0];

    if (!movups && opcode != 0x8B && opcode != 0x89 && opcode != 0x8D)
        return;

    auto modrm = movups ? code[2] : code[1];
    auto displacement = movups ? code + 3 : code + 2;
    auto mod = modrm >> 6;
    auto reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    auto rm = (modrm & 7) | ((rex & 1) << 3);
    auto wide = (rex & 8) != 0;

    if (opcode == 0x8D)
    {
        viewRegisters &= ~(1u << reg);
        return;
    }

    if (mod == 3 && !movups)
    {
        auto destination = opcode == 0x8B ? reg : rm;
        auto source = opcode == 0x8B ? rm : reg;

        if (wide && (viewRegisters & (1u << source)))
            viewRegisters |= 1u << destination;
        else
            viewRegisters &= ~(1u << destination);

        return;
    }

    // A store into memory changes no register
    if (opcode == 0x89)
        return;

    // [rm + disp32] without a SIB byte
    if (mod == 2 && (rm & 7) != 4 && (viewRegisters & (1u << rm)) && (movups || wide) && loadOffset == 0)
        loadOffset = ReadInt32(displacement);

    if (!movups)
        viewRegisters &= ~(1u << reg);
}

// What one instruction does to rdx, for telling AddPasses (FOutputs returned through the pointer in rdx, so it stores
// through rdx early) from the flattening (only reads its scene depth argument through rdx)
enum class RdxUse
{
    None,
    Store,       // writes memory through rdx
    Overwritten, // rdx no longer holds the second argument
    Stop         // call or ret, nothing after it tells
};

inline RdxUse ClassifyRdx(const uint8_t* code)
{
    constexpr uint8_t Rdx = 2;

    // Bounded, a prefix run longer than an instruction can be isn't code
    for (int i = 0; i < 4 && (*code == 0x66 || *code == 0xF2 || *code == 0xF3); i++)
        code++;

    uint8_t rex = 0;

    if ((*code & 0xF0) == 0x40)
        rex = *code++;

    auto twoByte = code[0] == 0x0F;
    auto opcode = twoByte ? code[1] : code[0];
    auto modrm = twoByte ? code[2] : code[1];
    auto mod = modrm >> 6;
    auto reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    auto rm = (modrm & 7) | ((rex & 1) << 3);

    // call rel32, call r/m, ret
    if (!twoByte && (opcode == 0xE8 || opcode == 0xC3 || (opcode == 0xFF && ((modrm >> 3) & 7) == 2)))
        return RdxUse::Stop;

    // mov r/m, r, mov r/m, imm, movups/movaps/movdqu m, xmm
    auto store = twoByte ? (opcode == 0x11 || opcode == 0x29 || opcode == 0x7F)
                         : (opcode == 0x88 || opcode == 0x89 || opcode == 0xC6 || opcode == 0xC7);

    // [rdx] or [rdx + disp], rm == 4 would be a SIB byte and rm == 5 with mod 0 rip relative, neither is rdx
    if (store && mod != 3 && rm == Rdx)
        return RdxUse::Store;

    // mov r, r/m, lea, mov edx, imm or xor edx, edx
    if (!twoByte && (((opcode == 0x8B || opcode == 0x8D) && reg == Rdx) || (opcode == 0xBA && (rex & 1) == 0) ||
                     ((opcode == 0x31 || opcode == 0x33) && mod == 3 && reg == Rdx)))
    {
        return RdxUse::Overwritten;
    }

    return RdxUse::None;
}

// mov [rsp + 20h], r64: the 5th argument of a call
inline bool StoresFifthArgument(const uint8_t* code)
{
    return (code[0] == 0x48 || code[0] == 0x4C) && code[1] == 0x89 && (code[2] & 0xC7) == 0x44 && code[3] == 0x24 &&
           code[4] == 0x20;
}

// mov byte ptr [rsp + 30h], 1: a true bool as the 7th argument of a call
inline bool SetsSeventhArgumentTrue(const uint8_t* code)
{
    return code[0] == 0xC6 && code[1] == 0x44 && code[2] == 0x24 && code[3] == 0x30 && code[4] == 1;
}
} // namespace XeSSUnrealDecode
