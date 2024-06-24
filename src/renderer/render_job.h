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
  PushConstants(const RenderConfig&, const Scene&);

  struct Viewport {
    glm::vec3 camera_position;
    alignas(16) glm::vec3 pixel_delta_u;
    alignas(16) glm::vec3 pixel_delta_v;
    alignas(16) glm::vec3 corner_pixel_pos;
  } viewport;

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

struct ImageCreateInfo {
  vk::Format format;
  std::uint32_t width, height;
  vk::ImageUsageFlags usage;
  vk::MemoryPropertyFlags memory_flags;
};

class RenderJob {
public:
  RenderJob(const Renderer&, const RenderConfig&, Scene&);
  
  auto render() const -> std::pair<const unsigned char*, std::size_t>;

private:
  void initialize_work_group_size();

  void process_scene();

  template <std::size_t index>
  void set_scene_data_storage_buffer_info(const auto&);
  template <std::size_t index>
  void set_uniform_buffer_info(const auto&);

  auto create_buffer(const BufferCreateInfo&) 
    -> std::unique_ptr<vk::raii::Buffer>;
  void create_buffers();
  void create_scene_data_storage_buffer();
  void create_uniform_buffer();
  void create_image_staging_buffer();
  void create_result_unstaging_buffer();

  auto create_image(const ImageCreateInfo&)
    -> vk::raii::Image;
  void create_images();
  
  void create_samplers();

  auto create_image_view(const vk::raii::Image&, vk::Format)
    -> vk::raii::ImageView;
  void create_image_views();

  void stage_scene_data_storage_buffer();
  void fill_uniform_buffer();
  void stage_images();

  void create_descriptor_set_layout();
  void create_descriptor_pool();
  void create_descriptor_set();
  void update_descriptor_set();

  void create_pipeline();

  void create_command_buffer();
  void record_command_buffer();
  void transition_images_for_usage();
  void copy_scene_data_to_device();
  void dispatch_compute_shader();
  void copy_result_to_host();

  const Renderer& renderer;
  const vk::raii::Device& device;
  const RenderConfig& config;
  VkAllocator allocator;

  Scene& scene;
  std::vector<BVHNode> bvh_nodes;
  std::vector<glm::vec4> vertex_positions;
  std::vector<PackedVertexData> packed_vertex_data;
  std::vector<PackedBVHNode> packed_bvh_nodes;

  std::uint32_t local_size_x, local_size_y;
  std::uint32_t bvh_max_depth;

  static constexpr std::uint32_t scene_data_storage_buffer_count = 3;
  std::array<BufferInfo, scene_data_storage_buffer_count> scene_data_storage_buffer_infos;

  static constexpr std::uint32_t uniform_buffer_count = 3;
  std::array<BufferInfo, uniform_buffer_count> uniform_buffer_infos;

  static constexpr std::uint32_t descriptor_count = scene_data_storage_buffer_count + uniform_buffer_count + 2;

  std::size_t scene_data_storage_buffer_size;
  std::unique_ptr<vk::raii::Buffer> scene_data_storage_buffer, scene_data_staging_buffer;

  std::size_t result_image_pixel_count;
  std::size_t result_unstaging_buffer_size;
  std::unique_ptr<vk::raii::Image> result_storage_image;
  std::unique_ptr<vk::raii::ImageView> result_storage_image_view;
  std::unique_ptr<vk::raii::Buffer> result_unstaging_buffer;

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

  static constexpr std::uint32_t specialization_constant_count = 6;
  using SpecializationConstant_t = std::uint32_t;
  std::array<SpecializationConstant_t, specialization_constant_count> specialization_constants;
  std::unique_ptr<vk::raii::PipelineLayout> pipeline_layout;
  std::unique_ptr<vk::raii::Pipeline> pipeline;

  std::unique_ptr<vk::raii::CommandPool> command_pool;
  std::unique_ptr<vk::raii::CommandBuffer> command_buffer;
};

template <std::size_t index>
void RenderJob::set_scene_data_storage_buffer_info(const auto& v) {
  scene_data_storage_buffer_infos[index] = BufferInfo {
    .ptr  = static_cast<const void*>(v.data()),
    .size = v.size() * sizeof(v[0])
  };
}

template <std::size_t index>
void RenderJob::set_uniform_buffer_info(const auto& v) {
  uniform_buffer_infos[index] = BufferInfo {
    .ptr  = static_cast<const void*>(v.data()),
    .size = v.size() * sizeof(v[0])
  };
}