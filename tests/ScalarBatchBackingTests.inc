// Optional span refusal must perform no transfers or observations. Exercise the
// production GPU-thread backing capability with protected pages and dirty bytes.
{
  const auto span_address = base + 0xa00;
  const std::array<uint32_t, 4> words{11, 22, 33, 44};
  for (uint32_t i = 0; i < words.size(); ++i) {
    Require(name, "clean span initialization",
            cache.TryWriteLabel(span_address + i * 4, &words[i], 4),
            "clean span initialization failed");
  }
  auto neighbor = cache.ObtainBuffer(base, 4, true, false);
  neighbor.first->Fill(neighbor.second, 4, pattern);
  const auto tick_before = scheduler.CurrentTick();
  std::array<uint32_t, 4> output{};
  Require(name, "clean span under dirty neighbor",
          Libs::LibKernel::Memory::TryReadCleanBufferBacking(
              span_address, output.data(), sizeof(output)) && output == words &&
              scheduler.CurrentTick() == tick_before && cache.HasGpuDirtyBytes(base, 4),
          "clean span refused or downloaded unrelated GPU bytes");
  auto overlapping = cache.ObtainBuffer(span_address + 4, 4, true, false);
  overlapping.first->Fill(overlapping.second, 4, 99);
  output.fill(0xdeadbeefu);
  const auto untouched = output;
  Require(name, "dirty span refusal",
          !Libs::LibKernel::Memory::TryReadCleanBufferBacking(
              span_address, output.data(), sizeof(output)) && output == untouched &&
              scheduler.CurrentTick() == tick_before &&
              cache.HasGpuDirtyBytes(span_address + 4, 4),
          "dirty span modified output, submitted or downloaded before fallback");
  // The ordinary/strict scalar fallback remains able to synchronize an actual
  // dirty DWORD after the optional capability declines the wider span.
  uint32_t scalar = 0;
  Require(name, "dirty span scalar fallback",
          Libs::LibKernel::Memory::TryReadBufferBacking(span_address + 4, &scalar, 4) &&
              scalar == 99, "declined span lost the normal synchronized scalar read");
  cache.ReadMemory(base, 0x1000);
  // A mapped prefix followed by an unmapped gap must not leak partial output.
  output.fill(0xdeadbeefu);
  const auto gap_tick = scheduler.CurrentTick();
  Require(name, "unmapped span refusal",
          !Libs::LibKernel::Memory::TryReadCleanBufferBacking(
              base + size - 4, output.data(), sizeof(output)) && output == untouched &&
              scheduler.CurrentTick() == gap_tick,
          "unavailable span copied its mapped prefix or submitted work");
  Require(name, "invalid span refusal",
          !Libs::LibKernel::Memory::TryReadCleanBufferBacking(UINT64_MAX - 3,
              output.data(), sizeof(output)) && output == untouched,
          "overflowing span modified output");
}
