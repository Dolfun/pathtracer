#pragma once
#include "render_config.h"
#include "vk_allocator.h"
#include <cstddef>
#include <vector>

class Renderer;

class RenderJob {
public:
  RenderJob(const Renderer&, const RenderConfig&);

  auto render() const -> std::vector<std::byte>;

private:
  const Renderer& renderer;
  RenderConfig config;
  VkAllocator allocator;
  std::size_t image_size;
};