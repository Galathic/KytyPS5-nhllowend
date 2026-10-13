#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// Reads `size` bytes at `address` from somewhere other than the faulting view; false when it
// cannot serve the read.
using LoadReader = bool (*)(uint64_t address, void* out, uint64_t size);

// Completes a faulting plain load (MOV/MOVZX/MOVSX(D) into a general register, or a whole-vector
// SSE/AVX load into XMM/YMM) with bytes from `reader`, then skips the instruction. Returns false
// without touching the context for anything else.
[[nodiscard]] bool TryEmulateLoad(void* native_context, LoadReader reader);

// Debugging: "<mnemonic> <prefix> mem=<bytes>" for the instruction at the context's RIP.
bool DescribeInstruction(void* native_context, char* out, uint32_t out_size);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
