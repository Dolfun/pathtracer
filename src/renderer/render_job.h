#pragma once
#include "render_config.h"
#include "vk_manager.h"
#include "vk_allocator.h"
#include <cstddef>

class RenderJob {
public:
  RenderJob(const VkManager&, const RenderConfig&);

  auto render() const -> std::vector<std::byte>;

private:
  const vk::raii::Device& device;
  VkAllocator vk_allocator;
  RenderConfig config;
  std::size_t image_size;
};