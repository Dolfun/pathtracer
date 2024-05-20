#pragma once
#include "vulkan_manager.h"

class Renderer {
public:
  Renderer();

private:
  VulkanManager vk_manager;
};