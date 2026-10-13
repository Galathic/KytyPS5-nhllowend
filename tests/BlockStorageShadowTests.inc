void CheckBlockStorageShadow() {
  constexpr const char* name = "BlockStorageShadow";
  auto& graphics = RuntimeContext();
  auto& renderer = RuntimeRenderer();
  for (const bool sixteen_bytes : {false, true}) {
    CommandScheduler scheduler(renderer, graphics);
    HW::Context registers{};
    HW::UserConfig user_config{};
    HW::Shader shaders{};
    scheduler.Begin(registers, user_config, shaders);
    ImageInfo info{};
    info.pixel_format = sixteen_bytes ? vk::Format::eBc6HUfloatBlock : vk::Format::eBc4UnormBlock;
    info.guest_format = sixteen_bytes ? Prospero::BufferFormat::kBc6UFloat : Prospero::BufferFormat::kBc4UNorm;
    info.extent = {8, 8, 1};
    info.resources = {2, 3};
    info.bytes_per_block = sixteen_bytes ? 16 : 8;
    Libs::Graphics::Image image(graphics, scheduler, info);
    // Exercise both block sizes even on hosts that support native BC4 storage.
    // Transfer usage is unchanged; this models the cache's unsupported-storage path.
    image.backing.usage &= ~vk::ImageUsageFlagBits::eStorage;
    Require(name, "shadow required", image.NeedsStorageShadow(),
        "test image did not enter the storage fallback path");
    const u32 words_per_block = info.bytes_per_block / sizeof(u32);
    std::vector<u32> expected(15 * words_per_block);
    for (u32 i = 0; i < expected.size(); ++i) expected[i] = 0x5a000000u + i;
    auto upload = CreateHostBuffer(name, expected.size() * sizeof(u32),
        vk::BufferUsageFlagBits::eTransferSrc, expected);
    auto readback = CreateHostBuffer(name, upload.size, vk::BufferUsageFlagBits::eTransferDst, {});
    std::array<vk::BufferImageCopy, 2> copies{};
    copies[0].imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 3};
    copies[0].imageExtent = {8, 8, 1};
    copies[1].bufferOffset = 12 * info.bytes_per_block;
    copies[1].imageSubresource = {vk::ImageAspectFlagBits::eColor, 1, 0, 3};
    copies[1].imageExtent = {4, 4, 1};
    image.Upload(copies, upload.buffer, 0, upload.size);
    // Change a single block in a selected mip/layer and verify every untouched
    // block. Reusing the same view also checks that the shadow is refreshed.
    for (u32 pass = 0; pass < 3; ++pass) {
      const u32 mip = pass == 1 ? 0 : 1;
      const u32 layer = pass == 1 ? 2 : 1;
      ImageViewInfo view{};
      view.format = sixteen_bytes ? vk::Format::eR32G32B32A32Uint : vk::Format::eR32G32Uint;
      view.type = vk::ImageViewType::e2DArray;
      view.aspect = vk::ImageAspectFlagBits::eColor;
      view.usage = vk::ImageUsageFlagBits::eStorage;
      view.base_level = mip;
      view.base_layer = layer;
      const auto native_view = image.PrepareShadowStorage(view);
      Require(name, "storage view", native_view != nullptr && image.PrepareShadowStorage(view) == native_view,
          "shadow storage view was not valid or reused");
      auto& shadow = ImageTestAccess::StorageShadow(image, mip);
      Require(name, "block grid", shadow.extent == vk::Extent3D{mip == 0 ? 2u : 1u, mip == 0 ? 2u : 1u, 1},
          "shadow dimensions did not use the selected mip's compressed block grid");
      std::vector<u32> replacement(words_per_block, 0x12340000u + pass);
      auto update = CreateHostBuffer(name, replacement.size() * sizeof(u32),
          vk::BufferUsageFlagBits::eTransferSrc, replacement);
      // The checks run without a GPU pipeline, so the recorder is direct.
      const auto recorder = scheduler.Current().Handle();
      const auto command  = recorder.Direct();
      ImageTestAccess::ShadowTransferWrite(image, mip, recorder);
      vk::BufferImageCopy write{};
      write.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
      write.imageOffset = {mip == 0 ? 1 : 0, 0, 0};
      write.imageExtent = {1, 1, 1};
      command.copyBufferToImage(update.buffer, shadow.image, vk::ImageLayout::eGeneral, write);
      image.FlushShadowStorage();
      image.FlushShadowStorage(); // A repeated flush must not copy stale state.
      const u32 block = mip == 0 ? 9 : 13;
      std::copy(replacement.begin(), replacement.end(), expected.begin() + block * words_per_block);
      image.Download(copies, readback.buffer, 0, readback.size);
      vk::BufferMemoryBarrier barrier{};
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = readback.buffer;
      barrier.size = readback.size;
      command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eHost, {}, {}, barrier, {});
      scheduler.Finish();
      Require(name, "raw blocks, untouched mips/layers and view reuse",
          ReadBuffer(name, readback, expected.size()) == expected,
          "shadow write changed the wrong block, mip or layer, or lost preserved bytes");
      DestroyBuffer(&update);
    }
    DestroyBuffer(&readback);
    DestroyBuffer(&upload);
    std::printf("[gpu]     %s %u-byte blocks: ok\n", name, info.bytes_per_block);
  }
}
