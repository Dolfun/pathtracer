#pragma once
#include <vector>
#include <cstddef>
#include "vk_manager.h"
#include "render_config.h"
#include <glm/vec3.hpp>

class Renderer {
public:
  Renderer();

  auto render(const RenderConfig&) const -> std::pair<const float*, std::size_t>;

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
  std::unique_ptr<vk::raii::Pipeline> pipeline;

  struct PushConstants {
    PushConstants(const RenderConfig&);

    struct Camera {
      glm::vec3 center;
      alignas(16) glm::vec3 pixel_delta_u;
      alignas(16) glm::vec3 pixel_delta_v;
      alignas(16) glm::vec3 corner_pixel_pos;
    } camera;

    std::uint32_t image_width, image_height;
    std::uint32_t seed;
    std::uint32_t nr_samples;
  };

  struct SpecializationConstants {
    std::uint32_t local_size_x;
    std::uint32_t local_size_y;
  } specialization_constants;
};