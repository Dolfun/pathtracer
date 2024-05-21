#pragma once
#include "render_config.h"
#include "vk_manager.h"
#include <vector>
#include <cstddef>

class RenderJob {
public:
  RenderJob(const VkManager&, const RenderConfig&);

  auto render() const -> std::vector<std::byte>;

private:
  const VkManager& vk_manager;
  RenderConfig config;
};