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
  void create_result_buffers();
  void update_descriptor_sets();
  void create_command_buffer();

  const Renderer& renderer;
  const vk::raii::Device& device;
  RenderConfig config;
  VkAllocator allocator;

  std::size_t buffer_size;
  std::unique_ptr<vk::raii::Buffer> result_buffer, result_staging_buffer;

  std::unique_ptr<vk::raii::CommandPool> command_pool;
  std::unique_ptr<vk::raii::CommandBuffer> command_buffer;
};