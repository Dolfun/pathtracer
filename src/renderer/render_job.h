#pragma once
#include "renderer.h"
#include "vk_allocator.h"
#include "../scene.h"
#include "bvh.h"

struct alignas(16) PackedVertexData {
  glm::vec4 normal_and_texcoord_u;
  glm::vec4 tangent_and_texcoord_v;
  glm::vec3 bitangent;
  std::int32_t material_index;
};
static_assert(alignof(PackedVertexData) == 16);
static_assert(sizeof(PackedVertexData)  == 48);

struct alignas(16) PackedBVHNode {
  glm::vec3 aabb_min;
  std::uint32_t left_or_begin;
  glm::vec3 aabb_max;
  std::uint32_t triangle_count;
};
static_assert(alignof(PackedBVHNode) == 16);
static_assert(sizeof(PackedBVHNode)  == 32);

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
  RenderJob(const Renderer&, const RenderConfig&, Scene&);
  
  auto render() const -> std::pair<const float*, std::size_t>;

private:
  void process_scene();

  template <std::size_t index>
  void add_input_buffer_info(const auto&);

  void create_input_buffers();
  void create_output_buffers();
  void create_images();
  void stage_input_buffer_data();
  void stage_image_data();
  void create_descriptor_set();
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

  static constexpr std::uint32_t input_descriptor_count = 4;
  std::array<InputBufferInfo, input_descriptor_count> input_buffer_infos;

  std::uint32_t image_count;
  std::size_t image_staging_buffer_size;
  std::vector<vk::raii::Image> images;
  std::unique_ptr<vk::raii::Buffer> image_staging_buffer;

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