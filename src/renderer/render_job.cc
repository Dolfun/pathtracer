#include "render_job.h"
#include "renderer.h"

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config)
  : renderer { _renderer }, device { renderer.device }, config { _config }, 
    allocator { renderer.device, renderer.vk_manager.get_physical_device().getMemoryProperties() } {
  
  result_pixel_count = config.image_width * config.image_height;
  buffer_size = result_pixel_count * NR_CHANNELS * sizeof(float);
  create_result_buffers();
  allocator.allocate_and_bind();
  update_descriptor_sets();
  create_command_buffer();

  command_buffer->reset();
  record_command_buffer();
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
    vk::MemoryPropertyFlagBits::eHostVisible  |
    vk::MemoryPropertyFlagBits::eHostCoherent |
    vk::MemoryPropertyFlagBits::eHostCached
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

void RenderJob::record_command_buffer() {
  vk::CommandBufferBeginInfo begin_info {
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  };
  command_buffer->begin(begin_info);

  command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *renderer.pipeline);
  command_buffer->bindDescriptorSets(
    vk::PipelineBindPoint::eCompute, *renderer.pipeline_layout, 0, { *renderer.descriptor_set }, {}
  );
  command_buffer->dispatch(result_pixel_count, 1, 1);

  vk::BufferMemoryBarrier2 buffer_memory_barrier {
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *result_buffer,
    .offset = 0,
    .size = buffer_size,
  };
  
  vk::DependencyInfo dependency_info {
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &buffer_memory_barrier,
  };
  command_buffer->pipelineBarrier2(dependency_info);

  vk::BufferCopy copy_region {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = buffer_size
  };
  command_buffer->copyBuffer(*result_buffer, *result_staging_buffer, { copy_region });

  command_buffer->end();
}

auto RenderJob::render() const -> std::pair<const float*, std::size_t> {
  vk::SubmitInfo submit_info {
    .commandBufferCount = 1,
    .pCommandBuffers = &(**command_buffer),
  };

  vk::FenceCreateInfo fence_create_info{};
  vk::raii::Fence fence { device, fence_create_info };
  renderer.vk_manager.get_compute_queue().submit(submit_info, fence);
  while (device.waitForFences({ fence }, true, UINT32_MAX) != vk::Result::eSuccess);

  auto bind_info = allocator.get_bind_info(*result_staging_buffer).value();
  void* ptr = (*device).mapMemory(bind_info.memory, 0, buffer_size);
  return std::make_pair(static_cast<const float*>(ptr), result_pixel_count * NR_CHANNELS);
}