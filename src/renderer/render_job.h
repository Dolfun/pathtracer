#pragma once
#include "renderer.h"
#include "vk_allocator.h"
#include "../scene/scene.h"

struct PushConstants {
  PushConstants(const RenderConfig&);

  struct Camera {
    glm::vec3 position;
    alignas(16) glm::vec3 pixel_delta_u;
    alignas(16) glm::vec3 pixel_delta_v;
    alignas(16) glm::vec3 corner_pixel_pos;
  } camera;

  std::uint32_t image_width, image_height;
  std::uint32_t seed;
  std::uint32_t sample_count;

  glm::vec3 bg_color;
};

struct SpecializationConstants {
  std::uint32_t local_size_x;
  std::uint32_t local_size_y;
};

struct InputBufferInfo {
  const void* ptr;
  std::size_t size;
};

class RenderJob {
public:
  RenderJob(const Renderer&, const RenderConfig&, OptimizedScene&);
  
  auto render() const -> std::pair<const float*, std::size_t>;

private:
  template <std::size_t index>
  void add_input_buffer_info(const auto&);

  void create_input_buffers();
  void create_output_buffers();
  void create_descriptor_set();
  void create_pipeline();
  void create_command_buffer();
  void record_command_buffer();

  const Renderer& renderer;
  const vk::raii::Device& device;
  const RenderConfig& config;
  VkAllocator allocator;
  OptimizedScene& scene;

  static constexpr std::uint32_t input_descriptor_count = 4;
  std::array<InputBufferInfo, input_descriptor_count> input_buffer_infos;

  std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout;
  std::unique_ptr<vk::raii::DescriptorPool> descriptor_pool;
  std::unique_ptr<vk::raii::DescriptorSet> descriptor_set;

  SpecializationConstants specialization_constants;
  std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout;
  std::unique_ptr<vk::raii::Pipeline> pipeline;

  std::unique_ptr<vk::raii::CommandPool> command_pool;
  std::unique_ptr<vk::raii::CommandBuffer> command_buffer;

  std::size_t input_buffer_size;
  std::unique_ptr<vk::raii::Buffer> input_buffer, input_staging_buffer;

  std::size_t output_image_pixel_count;
  std::size_t output_buffer_size;
  std::unique_ptr<vk::raii::Buffer> output_buffer, output_unstaging_buffer;
};

template <std::size_t index>
void RenderJob::add_input_buffer_info(const auto& v) {
  input_buffer_infos[index] = InputBufferInfo {
    .ptr  = static_cast<const void*>(v.data()),
    .size = v.size() * sizeof(v[0])
  };
}