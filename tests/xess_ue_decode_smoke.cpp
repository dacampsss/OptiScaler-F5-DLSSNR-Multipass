// x64 MSVC: cl /std:c++20 /EHsc /W4 /WX /O2 tests/xess_ue_decode_smoke.cpp
// The instruction forms inputs/XeSS_UE_Dx12.cpp reads in Intel's UE XeSS plugin to find its velocity flattening,
// View.ViewRect and GetSecondaryViewRectSize (OptiScaler/inputs/XeSS_UE_Decode.h). Each case is one real encoding.
#include "../OptiScaler/inputs/XeSS_UE_Decode.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

using namespace XeSSUnrealDecode;

static int failures = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

// Padded, the decoders may read a few bytes past a short instruction
static std::vector<uint8_t> Code(std::initializer_list<uint8_t> bytes)
{
    std::vector<uint8_t> code(bytes);
    code.resize(16, 0xCC);
    return code;
}

constexpr uint32_t Bit(int reg) { return 1u << reg; }
constexpr int Rax = 0, Rcx_ = 1, Rbx = 3, Rsi = 6, R9_ = 9, R14 = 14;

static void TrackCase(std::initializer_list<uint8_t> bytes, uint32_t before, uint32_t after, int32_t offset,
                      int line)
{
    auto code = Code(bytes);
    auto registers = before;
    int32_t loadOffset = 0;
    TrackViewRegisters(code.data(), registers, loadOffset);

    if (registers != after || loadOffset != offset)
    {
        printf("FAIL line %d: registers %X (want %X), offset %X (want %X)\n", line, registers, after, loadOffset,
               offset);
        ++failures;
    }
}
#define TRACK(bytes, before, after, offset) TrackCase(bytes, before, after, offset, __LINE__)
#define B(...) { __VA_ARGS__ }

int main()
{
    // --- calls ---
    {
        // call +0x10 from 0x1000, next instruction at 0x1005
        auto code = Code({ 0xE8, 0x10, 0x00, 0x00, 0x00 });
        auto call = DecodeCall(code.data(), 0x1005);
        CHECK(call.kind == Call::Kind::Direct && call.target == 0x1015);

        // call -0x10
        code = Code({ 0xE8, 0xF0, 0xFF, 0xFF, 0xFF });
        call = DecodeCall(code.data(), 0x1005);
        CHECK(call.kind == Call::Kind::Direct && call.target == 0x0FF5);

        // call qword ptr [rip + 0x200], with and without REX.W
        code = Code({ 0xFF, 0x15, 0x00, 0x02, 0x00, 0x00 });
        call = DecodeCall(code.data(), 0x2006);
        CHECK(call.kind == Call::Kind::Indirect && call.target == 0x2206);

        code = Code({ 0x48, 0xFF, 0x15, 0x00, 0x02, 0x00, 0x00 });
        call = DecodeCall(code.data(), 0x2007);
        CHECK(call.kind == Call::Kind::Indirect && call.target == 0x2207);

        // call rax isn't decoded
        code = Code({ 0xFF, 0xD0 });
        CHECK(DecodeCall(code.data(), 0x10).kind == Call::Kind::None);
    }

    // --- View register tracking, View starts in r9 (4th argument) ---
    // mov rbx, r9 (49 8B D9)
    TRACK(B(0x49, 0x8B, 0xD9), Bit(R9_), Bit(R9_) | Bit(Rbx), 0);
    // mov rcx, rbx (48 8B CB): View into the first argument
    TRACK(B(0x48, 0x8B, 0xCB), Bit(Rbx), Bit(Rbx) | Bit(Rcx_), 0);
    // mov rcx, rax (48 8B C8): rcx no longer View
    TRACK(B(0x48, 0x8B, 0xC8), Bit(Rcx_), 0, 0);
    // mov r14, r9 in the 89 form (4D 89 CE = mov r14, r9)
    TRACK(B(0x4D, 0x89, 0xCE), Bit(R9_), Bit(R9_) | Bit(R14), 0);
    // mov ecx, ebx (8B CB): a 32-bit copy zero-extends, rcx stops holding View
    TRACK(B(0x8B, 0xCB), Bit(Rbx) | Bit(Rcx_), Bit(Rbx), 0);
    // mov eax, dword ptr [r9 + 0x7A0] (41 8B 81 A0 07 00 00): ViewRect.Min.X read, eax overwritten
    TRACK(B(0x41, 0x8B, 0x81, 0xA0, 0x07, 0x00, 0x00), Bit(R9_) | Bit(Rax), Bit(R9_), 0);
    // mov rax, qword ptr [r9 + 0x7A0] (49 8B 81 A0 07 00 00): 64-bit load reports the offset
    TRACK(B(0x49, 0x8B, 0x81, 0xA0, 0x07, 0x00, 0x00), Bit(R9_), Bit(R9_), 0x7A0);
    // movups xmm0, [rbx + 0x8C0] (0F 10 83 C0 08 00 00): reports, View stays in rbx
    TRACK(B(0x0F, 0x10, 0x83, 0xC0, 0x08, 0x00, 0x00), Bit(Rbx), Bit(Rbx), 0x8C0);
    // movups xmm0, [rsi + 0x8C0] with rsi not View: nothing
    TRACK(B(0x0F, 0x10, 0x86, 0xC0, 0x08, 0x00, 0x00), Bit(Rbx), Bit(Rbx), 0);
    // mov rax, [rbx + 8] (48 8B 43 08) is disp8, not a ViewRect read
    TRACK(B(0x48, 0x8B, 0x43, 0x08), Bit(Rbx), Bit(Rbx), 0);
    // mov rax, [rsp + 0x100] (48 8B 84 24 00 01 00 00) has a SIB byte, not a View field
    TRACK(B(0x48, 0x8B, 0x84, 0x24, 0x00, 0x01, 0x00, 0x00), Bit(Rsi), Bit(Rsi), 0);
    // lea rcx, [rbx + 0x10] (48 8D 4B 10): rcx now a field address, not View
    TRACK(B(0x48, 0x8D, 0x4B, 0x10), Bit(Rbx) | Bit(Rcx_), Bit(Rbx), 0);
    // mov [rbx + 0x100], rax (48 89 83 00 01 00 00): a store changes nothing
    TRACK(B(0x48, 0x89, 0x83, 0x00, 0x01, 0x00, 0x00), Bit(Rbx), Bit(Rbx), 0);
    // A second load keeps the first offset
    {
        auto first = Code({ 0x49, 0x8B, 0x81, 0xA0, 0x07, 0x00, 0x00 });
        auto second = Code({ 0x49, 0x8B, 0x81, 0xB0, 0x07, 0x00, 0x00 });
        uint32_t registers = Bit(R9_);
        int32_t offset = 0;
        TrackViewRegisters(first.data(), registers, offset);
        TrackViewRegisters(second.data(), registers, offset);
        CHECK(offset == 0x7A0);
    }

    // --- rdx use: AddPasses stores its FOutputs through rdx, the flattening doesn't ---
    // mov qword ptr [rdx], rax (48 89 02)
    CHECK(ClassifyRdx(Code({ 0x48, 0x89, 0x02 }).data()) == RdxUse::Store);
    // mov qword ptr [rdx + 8], 0 (48 C7 42 08 00 00 00 00)
    CHECK(ClassifyRdx(Code({ 0x48, 0xC7, 0x42, 0x08, 0x00, 0x00, 0x00, 0x00 }).data()) == RdxUse::Store);
    // movups xmmword ptr [rdx], xmm0 (0F 11 02)
    CHECK(ClassifyRdx(Code({ 0x0F, 0x11, 0x02 }).data()) == RdxUse::Store);
    // movdqu xmmword ptr [rdx + 0x10], xmm1 (F3 0F 7F 4A 10)
    CHECK(ClassifyRdx(Code({ 0xF3, 0x0F, 0x7F, 0x4A, 0x10 }).data()) == RdxUse::Store);
    // mov qword ptr [r10], rax (49 89 02): r10, not rdx
    CHECK(ClassifyRdx(Code({ 0x49, 0x89, 0x02 }).data()) == RdxUse::None);
    // mov rax, qword ptr [rdx] (48 8B 02): a read through rdx, the flattening's scene depth argument
    CHECK(ClassifyRdx(Code({ 0x48, 0x8B, 0x02 }).data()) == RdxUse::None);
    // mov rdx, rbx (48 8B D3)
    CHECK(ClassifyRdx(Code({ 0x48, 0x8B, 0xD3 }).data()) == RdxUse::Overwritten);
    // lea rdx, [rsp + 0x40] (48 8D 54 24 40)
    CHECK(ClassifyRdx(Code({ 0x48, 0x8D, 0x54, 0x24, 0x40 }).data()) == RdxUse::Overwritten);
    // mov edx, 1 (BA 01 00 00 00); mov r10d, 1 (41 BA ...) is not rdx
    CHECK(ClassifyRdx(Code({ 0xBA, 0x01, 0x00, 0x00, 0x00 }).data()) == RdxUse::Overwritten);
    CHECK(ClassifyRdx(Code({ 0x41, 0xBA, 0x01, 0x00, 0x00, 0x00 }).data()) == RdxUse::None);
    // xor edx, edx (33 D2 and 31 D2)
    CHECK(ClassifyRdx(Code({ 0x33, 0xD2 }).data()) == RdxUse::Overwritten);
    CHECK(ClassifyRdx(Code({ 0x31, 0xD2 }).data()) == RdxUse::Overwritten);
    // call rel32, call rax, ret
    CHECK(ClassifyRdx(Code({ 0xE8, 0, 0, 0, 0 }).data()) == RdxUse::Stop);
    CHECK(ClassifyRdx(Code({ 0xFF, 0xD0 }).data()) == RdxUse::Stop);
    CHECK(ClassifyRdx(Code({ 0xC3 }).data()) == RdxUse::Stop);
    // push rbx (53), sub rsp, 0x40 (48 83 EC 40): prologue, nothing about rdx
    CHECK(ClassifyRdx(Code({ 0x53 }).data()) == RdxUse::None);
    CHECK(ClassifyRdx(Code({ 0x48, 0x83, 0xEC, 0x40 }).data()) == RdxUse::None);
    // A run of prefixes stays bounded
    CHECK(ClassifyRdx(Code({ 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66 }).data()) == RdxUse::None);

    // --- argument shapes of the flattening shared with XeFG ---
    // mov [rsp + 20h], rax (48 89 44 24 20); mov [rsp + 20h], r8 (4C 89 44 24 20)
    CHECK(StoresFifthArgument(Code({ 0x48, 0x89, 0x44, 0x24, 0x20 }).data()));
    CHECK(StoresFifthArgument(Code({ 0x4C, 0x89, 0x44, 0x24, 0x20 }).data()));
    // mov [rsp + 28h], rax is the 6th
    CHECK(!StoresFifthArgument(Code({ 0x48, 0x89, 0x44, 0x24, 0x28 }).data()));
    // mov byte ptr [rsp + 30h], 1 (C6 44 24 30 01); with 0 it is false
    CHECK(SetsSeventhArgumentTrue(Code({ 0xC6, 0x44, 0x24, 0x30, 0x01 }).data()));
    CHECK(!SetsSeventhArgumentTrue(Code({ 0xC6, 0x44, 0x24, 0x30, 0x00 }).data()));

    if (failures == 0)
        printf("xess_ue_decode_smoke: PASS\n");

    return failures == 0 ? 0 : 1;
}
