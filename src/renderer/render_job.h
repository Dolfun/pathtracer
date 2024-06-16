#pragma once
#include "renderer.h"
#include "vk_allocator.h"
#include "../scene.h"
#include "bvh.h"

struct alignas(16) PackedVertexData {
  glm::vec4 position;
  glm::vec4 normal_and_texcoord_u;
  glm::vec4 tangent_and_texcoord_v;
  glm::vec3 bitangent;
  std::int32_t material_index;
};

struct alignas(16) PackedBVHNode {
  glm::vec3 aabb_min;
  std::uint32_t left_or_begin;
  glm::vec3 aabb_max;
  std::uint32_t triangle_count;
};

struct PushConstants {
  PushConstants(const RenderConfig&);

  struct Camera {
    glm::vec3 position;
    alignas(16) glm::vec3 pixel_delta_u;
    alignas(16) glm::vec3 pixel_delta_v;
    alignas(16) glm::vec3 corner_pixel_pos;
  } camera;

  std::uint32_t resolution_x, resolution_y;
  std::uint32_t seed;
  std::uint32_t sample_count;

  glm::vec3 bg_color;
};

struct BufferInfo {
  const void* ptr;
  std::size_t size;
};

struct BufferCreateInfo {
  std::size_t size;
  vk::BufferUsageFlags usage;
  vk::MemoryPropertyFlags memory_flags;
};

class RenderJob {
public:
  RenderJob(const Renderer&, const RenderConfig&, Scene&);
  
  auto render() const -> std::pair<const float*, std::size_t>;

private:
  void process_scene();

  template <std::size_t index>
  void add_input_storage_buffer_info(const auto&);
  template <std::size_t index>
  void add_uniform_buffer_info(const auto&);

  auto create_buffer(const BufferCreateInfo&) 
    -> std::unique_ptr<vk::raii::Buffer>;
  void create_buffers();
  void create_images();
  void create_samplers();
  void create_image_views();

  void stage_input_storage_buffer_data();
  void copy_uniform_buffer_data();
  void stage_image_data();

  void create_descriptor_set_layout();
  void create_descriptor_pool();
  void create_descriptor_set();
  void update_descriptor_set();

  void create_pipeline();

  void create_command_buffer();
  void record_command_buffer();
  void transition_images_for_copy();
  void copy_input_resources();
  void dispatch();
  void copy_output_resources();

  const Renderer& renderer;
  const vk::raii::Device& device;
  const RenderConfig& config;
  VkAllocator allocator;

  Scene& scene;
  std::vector<BVHNode> bvh_nodes;
  std::vector<glm::vec4> vertex_positions;
  std::vector<PackedVertexData> packed_vertex_data;
  std::vector<PackedBVHNode> packed_bvh_nodes;

  static constexpr std::uint32_t input_storage_buffer_count = 3;
  std::array<BufferInfo, input_storage_buffer_count> input_storage_buffer_infos;

  static constexpr std::uint32_t uniform_buffer_count = 3;
  std::array<BufferInfo, uniform_buffer_count> uniform_buffer_infos;

  static constexpr std::uint32_t storage_buffer_count = input_storage_buffer_count + 1;
  static constexpr std::uint32_t descriptor_count = storage_buffer_count + uniform_buffer_count + 1;

  std::size_t input_storage_buffer_size;
  std::unique_ptr<vk::raii::Buffer> input_storage_buffer, input_staging_buffer;

  std::size_t output_image_pixel_count;
  std::size_t output_storage_buffer_size;
  std::unique_ptr<vk::raii::Buffer> output_storage_buffer, output_unstaging_buffer;

  std::size_t uniform_buffer_size;
  std::unique_ptr<vk::raii::Buffer> uniform_buffer;

  std::uint32_t image_count;
  std::uint32_t combined_image_sampler_count;
  std::size_t image_staging_buffer_size;
  std::vector<vk::raii::Image> images;
  std::vector<vk::raii::Sampler> samplers;
  std::vector<vk::raii::ImageView> image_views;
  std::unique_ptr<vk::raii::Buffer> image_staging_buffer;

  std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout;
  std::unique_ptr<vk::raii::DescriptorPool> descriptor_pool;
  std::unique_ptr<vk::raii::DescriptorSet> descriptor_set;

  static constexpr std::uint32_t specialization_constant_count = 5;
  using SpecializationConstant_t = std::uint32_t;
  std::array<SpecializationConstant_t, specialization_constant_count> specialization_constants;
  std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout;
  std::unique_ptr<vk::raii::Pipeline> pipeline;

  std::unique_ptr<vk::raii::CommandPool> command_pool;
  std::unique_ptr<vk::raii::CommandBuffer> command_buffer;
};

template <std::size_t index>
void RenderJob::add_input_storage_buffer_info(const auto& v) {
  input_storage_buffer_infos[index] = BufferInfo {
    .ptr  = static_cast<const void*>(v.data()),
    .size = v.size() * sizeof(v[0])
  };
}

template <std::size_t index>
void RenderJob::add_uniform_buffer_info(const auto& v) {
  uniform_buffer_infos[index] = BufferInfo {
    .ptr  = static_cast<const void*>(v.data()),
    .size = v.size() * sizeof(v[0])
  };
}