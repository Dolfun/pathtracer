#pragma once
#include <vector>
#include <cstddef>
#include "vk_manager.h"
#include "render_config.h"

class Renderer {
public:
  Renderer();

  auto render(const RenderConfig&) const -> std::vector<std::byte>;

  friend class RenderJob;

private:
  auto read_binary_file(const std::string&) -> std::vector<std::byte>;

  void create_descriptors();
  void create_compute_pipeline();

  VkManager vk_manager;
  const vk::raii::Device& device;

  std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout;
  std::unique_ptr<vk::raii::DescriptorPool> descriptor_pool;
  std::unique_ptr<vk::raii::DescriptorSet> descriptor_set;

  std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout;
  std::unique_ptr<vk::raii::Pipeline> compute_pipeline;
};