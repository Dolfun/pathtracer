#include "render_job.h"
#include "renderer.h"
#include <fstream>
#include <fmt/core.h>
#include <glm/glm.hpp>

auto read_binary_file(const std::string& path) -> std::vector<std::byte>;

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config, const Scene& _scene)
  : renderer { _renderer }, device { *renderer.device }, config { _config }, scene { _scene },
    allocator { *renderer.device, renderer.physical_device->getMemoryProperties() } {
  
  create_scene_buffers();
  create_result_buffers();
  allocator.allocate_and_bind();

  create_descriptor_set();
  create_pipeline();

  create_command_buffer();
  command_buffer->reset();
  record_command_buffer();
}

void RenderJob::create_scene_buffers() {
  scene_buffer_size = scene.triangles.size() * sizeof(Scene::Triangle);

  vk::BufferCreateInfo scene_staging_buffer_create_info {
    .size = scene_buffer_size,
    .usage = vk::BufferUsageFlagBits::eTransferSrc,
    .sharingMode = vk::SharingMode::eExclusive
  };
  scene_staging_buffer = std::make_unique<vk::raii::Buffer>(device, scene_staging_buffer_create_info);
  allocator.add_resource(*scene_staging_buffer, 
    vk::MemoryPropertyFlagBits::eHostVisible |
    vk::MemoryPropertyFlagBits::eHostCoherent
  );

  vk::BufferCreateInfo scene_buffer_create_info {
    .size = scene_buffer_size,
    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .sharingMode = vk::SharingMode::eExclusive
  };
  scene_buffer = std::make_unique<vk::raii::Buffer>(device, scene_buffer_create_info);
  allocator.add_resource(*scene_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void RenderJob::create_result_buffers() {
  result_image_pixel_count = config.image_width * config.image_height;
  result_buffer_size = result_image_pixel_count * NR_CHANNELS * sizeof(float);

  vk::BufferCreateInfo result_buffer_create_info {
    .size = result_buffer_size,
    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
    .sharingMode = vk::SharingMode::eExclusive
  };
  result_buffer = std::make_unique<vk::raii::Buffer>(device, result_buffer_create_info);
  allocator.add_resource(*result_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

  vk::BufferCreateInfo result_staging_buffer_create_info {
    .size = result_buffer_size,
    .usage = vk::BufferUsageFlagBits::eTransferDst,
    .sharingMode = vk::SharingMode::eExclusive
  };
  result_unstaging_buffer = std::make_unique<vk::raii::Buffer>(device, result_staging_buffer_create_info);
  allocator.add_resource(
    *result_unstaging_buffer, 
    vk::MemoryPropertyFlagBits::eHostVisible  |
    vk::MemoryPropertyFlagBits::eHostCoherent |
    vk::MemoryPropertyFlagBits::eHostCached
  );
}

void RenderJob::create_descriptor_set() {
  // Descriptor set layout
  constexpr std::uint32_t descriptor_count = 2;
  std::array<vk::DescriptorSetLayoutBinding, descriptor_count> layout_bindings{};
  for (std::uint32_t i = 0; i < descriptor_count; ++i) {
    layout_bindings[i] = {
      .binding = i,
      .descriptorType = vk::DescriptorType::eStorageBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eCompute
    };
  }

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .bindingCount = static_cast<std::uint32_t>(layout_bindings.size()),
    .pBindings = layout_bindings.data()
  };
  descriptor_set_layout = 
    std::make_unique<vk::raii::DescriptorSetLayout>(device, descriptor_set_layout_create_info);

  // Descriptor pool
  vk::DescriptorPoolSize pool_size {
    .type = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = descriptor_count
  };

  vk::DescriptorPoolCreateInfo pool_create_info {
    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    .maxSets = 1,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  descriptor_pool = std::make_unique<vk::raii::DescriptorPool>(device, pool_create_info);

  // Descriptor set
  vk::DescriptorSetAllocateInfo allocate_info {
    .descriptorPool = *descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &(**descriptor_set_layout)
  };
  vk::raii::DescriptorSets descriptor_sets { device, allocate_info };
  descriptor_set = std::make_unique<vk::raii::DescriptorSet>(std::move(descriptor_sets.front()));

  // Descriptor Writes
  std::array buffers { **scene_buffer, **result_buffer };
  std::array buffer_sizes { scene_buffer_size, result_buffer_size };
  std::vector<vk::DescriptorBufferInfo> buffer_infos;
  std::vector<vk::WriteDescriptorSet> descriptor_writes;
  buffer_infos.resize(descriptor_count);
  descriptor_writes.resize(descriptor_count);
  for (std::uint32_t i = 0; i < descriptor_count; ++i) {
    buffer_infos[i] = {
      .buffer = buffers[i],
      .offset = 0,
      .range = buffer_sizes[i]
    };

    descriptor_writes[i] = {
      .dstSet = *descriptor_set,
      .dstBinding = i,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = vk::DescriptorType::eStorageBuffer,
      .pBufferInfo = &buffer_infos[i]
    };
  }
  device.updateDescriptorSets(descriptor_writes, {});
}

void RenderJob::create_pipeline() {
  // Shader Stage
  auto compute_shader_code = read_binary_file("shaders/main.comp.spv");
  vk::ShaderModuleCreateInfo shader_module_create_info {
    .codeSize = compute_shader_code.size(),
    .pCode = reinterpret_cast<const std::uint32_t*>(compute_shader_code.data())
  };
  vk::raii::ShaderModule shader_module { device, shader_module_create_info };

  vk::SpecializationMapEntry specialization_map_entries[2];
  specialization_map_entries[0] = vk::SpecializationMapEntry {
    .constantID = 0,
    .offset = offsetof(SpecializationConstants, local_size_x),
    .size = sizeof(SpecializationConstants::local_size_x)
  };
  specialization_map_entries[1] = vk::SpecializationMapEntry {
    .constantID = 1,
    .offset = offsetof(SpecializationConstants, local_size_y),
    .size = sizeof(SpecializationConstants::local_size_y)
  };

  specialization_constants = {
    .local_size_x = 32,
    .local_size_y = 32
  };

  vk::SpecializationInfo specialization_info {
    .mapEntryCount = 2,
    .pMapEntries = specialization_map_entries,
    .dataSize = sizeof(SpecializationConstants),
    .pData = &specialization_constants
  };

  vk::PipelineShaderStageCreateInfo shader_stage_create_info {
    .stage = vk::ShaderStageFlagBits::eCompute,
    .module = shader_module,
    .pName = "main",
    .pSpecializationInfo = &specialization_info
  };

  // Push Constant
  vk::PushConstantRange push_constant_range {
    .stageFlags = vk::ShaderStageFlagBits::eCompute,
    .offset = 0,
    .size = sizeof(PushConstants)
  };

  // Pipeline layout
  vk::PipelineLayoutCreateInfo layout_create_info {
    .setLayoutCount = 1,
    .pSetLayouts = &(**descriptor_set_layout),
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_constant_range
  };
  pipeline_layout = std::make_unique<vk::raii::PipelineLayout>(device, layout_create_info);

  // Pipeline
  vk::ComputePipelineCreateInfo create_info {
    .stage = shader_stage_create_info,
    .layout = *pipeline_layout
  };
  pipeline = std::make_unique<vk::raii::Pipeline>(device, nullptr, create_info);
}

void RenderJob::create_command_buffer() {
  vk::CommandPoolCreateInfo pool_create_info {
    .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
    .queueFamilyIndex = renderer.compute_family_index.value()
  };
  command_pool = std::make_unique<vk::raii::CommandPool>(device, pool_create_info);

  vk::CommandBufferAllocateInfo allocate_info {
    .commandPool = *command_pool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1
  };
  vk::raii::CommandBuffers command_buffers { device, allocate_info };
  command_buffer = std::make_unique<vk::raii::CommandBuffer>(std::move(command_buffers.front()));
}

void RenderJob::record_command_buffer() {
  vk::CommandBufferBeginInfo begin_info {
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  };
  command_buffer->begin(begin_info);

  command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
  command_buffer->bindDescriptorSets(
    vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, { *descriptor_set }, {}
  );

  PushConstants push_constants { config, scene };
  command_buffer->pushConstants<PushConstants>(
    *pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, { push_constants }
  );

  vk::BufferCopy scene_buffer_copy_info {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = scene_buffer_size
  };
  command_buffer->copyBuffer(
    *scene_staging_buffer, *scene_buffer, { scene_buffer_copy_info }
  );

  vk::BufferMemoryBarrier2 scene_buffer_memory_barrier {
    .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *scene_buffer,
    .offset = 0,
    .size = scene_buffer_size
  };

  vk::DependencyInfo scene_buffer_dependency_info {
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &scene_buffer_memory_barrier
  };
  command_buffer->pipelineBarrier2(scene_buffer_dependency_info);

  auto [local_size_x, local_size_y] = specialization_constants;
  std::uint32_t global_size_x = (config.image_height + local_size_x - 1) / local_size_x;
  std::uint32_t global_size_y = (config.image_width  + local_size_y - 1) / local_size_y;
  command_buffer->dispatch(global_size_x, global_size_y, 1);

  vk::BufferMemoryBarrier2 result_buffer_memory_barrier {
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *result_buffer,
    .offset = 0,
    .size = result_buffer_size
  };
  
  vk::DependencyInfo result_buffer_dependency_info {
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &result_buffer_memory_barrier,
  };
  command_buffer->pipelineBarrier2(result_buffer_dependency_info);

  vk::BufferCopy result_buffer_copy_info {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = result_buffer_size
  };
  command_buffer->copyBuffer(
    *result_buffer, *result_unstaging_buffer, { result_buffer_copy_info }
  );

  command_buffer->end();
}

auto RenderJob::render() const -> std::pair<const float*, std::size_t> {
  {
    auto bind_info = allocator.get_bind_info(*scene_staging_buffer).value();
    void* ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, scene_buffer_size);
    std::memcpy(ptr, scene.triangles.data(), scene_buffer_size);
  }

  vk::SubmitInfo submit_info {
    .commandBufferCount = 1,
    .pCommandBuffers = &(**command_buffer),
  };

  vk::FenceCreateInfo fence_create_info{};
  vk::raii::Fence fence { device, fence_create_info };
  renderer.compute_queue->submit(submit_info, fence);
  while (device.waitForFences({ fence }, true, UINT32_MAX) != vk::Result::eSuccess) {};

  {
    auto bind_info = allocator.get_bind_info(*result_unstaging_buffer).value();
    void* ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, result_buffer_size);
    return std::make_pair(static_cast<const float*>(ptr), result_image_pixel_count * NR_CHANNELS);
  }
}

PushConstants::PushConstants(const RenderConfig& config, const Scene& scene) :
  image_width { config.image_width }, image_height { config.image_height },
  seed { config.seed }, sample_count { config.sample_count },
  triangle_count { static_cast<std::uint32_t>(scene.triangles.size()) } {

  float image_width = static_cast<float>(config.image_width);
  float image_height = static_cast<float>(config.image_height);
  const auto& camera = config.camera;

  glm::vec3 w = glm::normalize(camera.center - camera.lookat);
  glm::vec3 u = glm::normalize(glm::cross(camera.up, w));
  glm::vec3 v = glm::cross(w, u);

  float focal_length = glm::length(camera.center - camera.lookat);
  float theta = glm::radians(camera.vertical_fov);
  float viewport_height = 2.0f * glm::tan(theta / 2.0f) * focal_length;
  float viewport_width = viewport_height * image_width / image_height;

  glm::vec3 viewport_u = viewport_width * u;
  glm::vec3 viewport_v = viewport_height * -v;
  glm::vec3 pixel_delta_u = viewport_u / image_width;
  glm::vec3 pixel_delta_v = viewport_v / image_height;
  glm::vec3 viewport_upper_left = camera.center - focal_length * w - 0.5f * (viewport_u + viewport_v);
  glm::vec3 corner_pixel_pos = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

  this->camera = {
    .center = camera.center,
    .pixel_delta_u = pixel_delta_u,
    .pixel_delta_v = pixel_delta_v,
    .corner_pixel_pos = corner_pixel_pos
  };
}

auto read_binary_file(const std::string& path) -> std::vector<std::byte> {
  std::ifstream file { path, std::ios::ate | std::ios::binary };
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  auto size = static_cast<size_t>(file.tellg());
  std::vector<std::byte> buffer(size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  
  return buffer;
}