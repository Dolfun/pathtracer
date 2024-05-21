#pragma once
#include <vector>
#include <cstddef>
#include "vk_manager.h"
#include "render_config.h"

class Renderer {
public:
  auto render(const RenderConfig&) const -> std::vector<std::byte>;

private:
  VkManager vk_manager;
};