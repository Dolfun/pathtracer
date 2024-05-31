#pragma once
#include "renderer.h"
#include "vk_allocator.h"

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
};

class RenderJob {
public:
  RenderJob(const Renderer&, const RenderConfig&);
  auto render() const -> std::pair<const float*, std::size_t>;

private:
  void create_descriptor_set();
  void create_pipeline();
  void create_result_buffers();
  void create_command_buffer();
  void update_descriptor_sets();
  void record_command_buffer();

  const Renderer& renderer;
  const vk::raii::Device& device;
  const RenderConfig& config;
  VkAllocator allocator;

  std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout;
  std::unique_ptr<vk::raii::DescriptorPool> descriptor_pool;
  std::unique_ptr<vk::raii::DescriptorSet> descriptor_set;

  SpecializationConstants specialization_constants;
  std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout;
  std::unique_ptr<vk::raii::Pipeline> pipeline;

  std::unique_ptr<vk::raii::CommandPool> command_pool;
  std::unique_ptr<vk::raii::CommandBuffer> command_buffer;

  std::size_t result_image_pixel_count;
  std::size_t buffer_size;
  std::unique_ptr<vk::raii::Buffer> result_buffer, result_unstaging_buffer;
};