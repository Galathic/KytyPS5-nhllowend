// Included inside CheckInlineLabelWrites' GPU-thread command. Uses the real
// CommandProcessor and BufferCache, including protected pages and GPU readback.
if (Common::RuntimeFlags::Current().pm4_write_data_fast) {
  const std::array<uint32_t, 3> input{0x12345678u, 0x87654321u, 0xabcdef01u};
  const auto fill_neighbor = [&] {
    auto dirty = cache.ObtainBuffer(base, 4, true, false);
    dirty.first->Fill(dirty.second, 4, pattern);
  };
  const auto read_word = [&](uint64_t address) {
    uint32_t word = 0;
    Require(name, "WRITE_DATA backing read",
            Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word)),
            "WRITE_DATA backing unavailable");
    return word;
  };
  // Contiguous 4/8-byte writes and repeated-address stores must retain adjacent
  // GPU ownership, avoid queue submission, and survive a later GPU download.
  for (const auto count : {1u, 2u}) {
    for (const bool one_address : {false, true}) {
      fill_neighbor();
      const auto destination = base + 0x500;
      const auto tick_before = scheduler.CurrentTick();
      processor.WriteData(reinterpret_cast<uint32_t*>(destination), input.data(), count,
                          one_address ? (1u << 16u) : 0u);
      Require(name, "WRITE_DATA no submission",
              scheduler.CurrentTick() == tick_before && cache.HasGpuDirtyBytes(base, 4),
              "small WRITE_DATA submitted or lost neighboring GPU ownership");
      Require(name, "WRITE_DATA immediate result",
              read_word(destination) == input[one_address ? count - 1u : 0u] &&
                  (one_address || count == 1 || read_word(destination + 4) == input[1]),
              "small WRITE_DATA produced wrong backing values");
      cache.ReadMemory(base, 0x1000);
      Require(name, "WRITE_DATA download coherence",
              read_word(base) == pattern &&
                  read_word(destination) == input[one_address ? count - 1u : 0u] &&
                  (one_address || count == 1 || read_word(destination + 4) == input[1]),
              "download restored stale WRITE_DATA values or damaged GPU neighbor");
    }
  }
  // The third source overlaps the repeatedly written destination. Reading all
  // values before the loop would yield input[2]; the second scalar store writes
  // input[1] into the third source before it is read, so original ordering yields input[1].
  const auto aliased_source = base + 0x600;
  for (uint32_t i = 0; i < input.size(); ++i) {
    Require(name, "WRITE_DATA alias initialization",
            cache.TryWriteLabel(aliased_source + i * 4, &input[i], 4),
            "alias initialization failed");
  }
  processor.WriteData(reinterpret_cast<uint32_t*>(aliased_source + 8),
                      reinterpret_cast<const uint32_t*>(aliased_source), 3, 1u << 16u);
  Require(name, "WRITE_DATA alias order", read_word(aliased_source + 8) == input[1],
          "repeated-address WRITE_DATA snapshotted or reordered aliased source");
  cache.ReadMemory(base, 0x1000);
  Require(name, "WRITE_DATA alias download", read_word(aliased_source + 8) == input[1],
          "alias result was lost on GPU download");
  // Unsupported cache destination retains normal host-memory fallback.
  std::array<uint32_t, 3> host_output{};
  processor.WriteData(host_output.data(), input.data(), 2, 0);
  Require(name, "WRITE_DATA host fallback", host_output[0] == input[0] &&
              host_output[1] == input[1], "WRITE_DATA host fallback changed");
  // Larger contiguous packets use the unchanged memcpy path.
  processor.WriteData(host_output.data(), input.data(), 3, 0);
  Require(name, "WRITE_DATA larger fallback", host_output == input,
          "larger WRITE_DATA fallback changed");
  processor.WriteData(host_output.data(), input.data(), 3, 1u << 16u);
  Require(name, "WRITE_DATA repeated host fallback", host_output[0] == input[2] &&
              host_output[1] == input[1], "repeated host fallback changed ordering or neighbors");
  processor.WriteData(nullptr, nullptr, 0, 0);
}

if (Common::RuntimeFlags::Current().pm4_event_clean) {
  // Clean EVENT_WRITE payload on a page made dirty solely by an unrelated neighbor. The
  // payload fast path does not depend on KYTY_PERF_NO_SYNC_READBACK and is on by default,
  // so this coverage runs without any environment setup.
  // The parser itself is called so header-fetch policy does not hide payload behavior.
  const uint32_t event = 0x16u | (7u << 8u);
  const auto event_payload = base + 0x700;
  Require(name, "EVENT_WRITE initialization", cache.TryWriteLabel(event_payload, &event, 4),
          "EVENT_WRITE initialization failed");
  auto dirty = cache.ObtainBuffer(base, 4, true, false);
  dirty.first->Fill(dirty.second, 4, pattern);
  const auto tick_before = scheduler.CurrentTick();
  Require(name, "EVENT_WRITE clean payload",
          g_cp_op_func[Pm4::IT_EVENT_WRITE](processor,
              KYTY_PM4(2, Pm4::IT_EVENT_WRITE, 0),
              reinterpret_cast<const uint32_t*>(event_payload), 2, 2) == 1 &&
              scheduler.CurrentTick() == tick_before && cache.HasGpuDirtyBytes(base, 4),
          "EVENT_WRITE decoded incorrectly or downloaded dirty neighbor");
  // A truly GPU-dirty event DWORD must reject peeking and download the actual
  // GPU value. CPU backing contains the preceding different event/index.
  auto dirty_event = cache.ObtainBuffer(event_payload, 4, true, false);
  dirty_event.first->Fill(dirty_event.second, 4, 0x2au);
  uint32_t peek = 0;
  Require(name, "EVENT_WRITE dirty refusal",
          !processor.TryPeekGuest(reinterpret_cast<const void*>(event_payload), &peek, 4),
          "dirty event payload was exposed through clean backing");
  Require(name, "EVENT_WRITE dirty fallback",
          g_cp_op_func[Pm4::IT_EVENT_WRITE](processor,
              KYTY_PM4(2, Pm4::IT_EVENT_WRITE, 0),
              reinterpret_cast<const uint32_t*>(event_payload), 2, 2) == 1,
          "dirty EVENT_WRITE parser failed");
  uint32_t downloaded = 0;
  Require(name, "EVENT_WRITE actual GPU value",
          Libs::LibKernel::Memory::TryReadBacking(event_payload, &downloaded, 4) &&
              downloaded == 0x2au, "dirty EVENT_WRITE used stale CPU backing");
  // Addressed event must keep its result destination, not redirect writes into
  // the local clean metadata snapshot. Host result array avoids unrelated GPU faults.
  alignas(8) std::array<uint64_t, 32> results{};
  const uint64_t result_address = reinterpret_cast<uint64_t>(results.data());
  const std::array<uint32_t, 3> addressed{0x139u,
      static_cast<uint32_t>(result_address), static_cast<uint32_t>(result_address >> 32u)};
  for (uint32_t i = 0; i < addressed.size(); ++i) {
    Require(name, "EVENT_WRITE address initialization",
            cache.TryWriteLabel(event_payload + i * 4, &addressed[i], 4),
            "EVENT_WRITE address initialization failed");
  }
  dirty = cache.ObtainBuffer(base, 4, true, false);
  dirty.first->Fill(dirty.second, 4, pattern);
  const auto address_tick = scheduler.CurrentTick();
  Require(name, "EVENT_WRITE addressed payload",
          g_cp_op_func[Pm4::IT_EVENT_WRITE](processor,
              KYTY_PM4(4, Pm4::IT_EVENT_WRITE, 0),
              reinterpret_cast<const uint32_t*>(event_payload), 4, 4) == 3 &&
              scheduler.CurrentTick() == address_tick && cache.HasGpuDirtyBytes(base, 4),
          "addressed EVENT_WRITE downloaded unrelated GPU data");
  for (uint32_t i = 0; i < 16; ++i) {
    Require(name, "EVENT_WRITE preserved target", (results[i * 2] >> 63u) == 1 &&
                results[i * 2 + 1] == 0, "addressed event lost target or wrote odd counter");
  }
  cache.ReadMemory(base, 0x1000);
}
