#include "render_job.h"
#include "renderer.h"

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config)
  : renderer { _renderer }, device { renderer.device }, config { _config }, 
    allocator { renderer.device, renderer.vk_manager.get_physical_device().getMemoryProperties() } {
  
  buffer_size = config.image_width * config.image_height * NR_CHANNELS * sizeof(uint32_t);
  create_result_buffers();
  allocator.allocate_and_bind();
  update_descriptor_sets();
  create_command_buffer();
}

void RenderJob::create_result_buffers() {
  vk::BufferCreateInfo result_buffer_create_info {
    .size = buffer_size,
    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc
  };
  result_buffer = std::make_unique<vk::raii::Buffer>(device, result_buffer_create_info);
  allocator.add_resource(*result_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

  vk::BufferCreateInfo result_staging_buffer_create_info {
    .size = buffer_size,
    .usage = vk::BufferUsageFlagBits::eTransferDst
  };
  result_staging_buffer = std::make_unique<vk::raii::Buffer>(device, result_staging_buffer_create_info);
  allocator.add_resource(
    *result_staging_buffer, 
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
  );
}

void RenderJob::update_descriptor_sets() {
  vk::DescriptorBufferInfo buffer_info {
    .buffer = *result_buffer,
    .offset = 0,
    .range = buffer_size
  };

  vk::WriteDescriptorSet descriptor_write {
    .dstSet = *renderer.descriptor_set,
    .dstBinding = 0,
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType = vk::DescriptorType::eStorageBuffer,
    .pBufferInfo = &buffer_info
  };

  device.updateDescriptorSets({ descriptor_write }, {});
}

void RenderJob::create_command_buffer() {
  vk::CommandPoolCreateInfo pool_create_info {
    .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
    .queueFamilyIndex = renderer.vk_manager.get_queue_family_index()
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

auto RenderJob::render() const -> std::vector<std::byte> {
  std::vector<std::byte> result(buffer_size);

  return result;
}