#include "render_job.h"
#include "renderer.h"
#include "../timeit.h"
#include <fstream>
#include <glm/glm.hpp>

constexpr std::uint32_t local_size_x = 32;
constexpr std::uint32_t local_size_y = 32;

auto read_binary_file(const std::string& path) -> std::vector<std::byte>;
auto get_filter_type (Scene::SamplerFilter_t filter) -> vk::Filter;
auto get_wrap_type(Scene::SamplerWarp_t wrap) -> vk::SamplerAddressMode;

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config, Scene& _scene)
  : renderer { _renderer }, device { *renderer.device }, config { _config },
    allocator { *renderer.device, renderer.physical_device->getMemoryProperties() },
    scene { _scene } {

  timeit("Building BVH", [&] {
    bvh_nodes = build_bvh(scene, 16);
  });

  process_scene();

  add_input_storage_buffer_info<0>(vertex_positions);
  add_input_storage_buffer_info<1>(packed_vertex_data);
  add_input_storage_buffer_info<2>(packed_bvh_nodes);
  
  add_uniform_buffer_info<0>(scene.materials);
  
  create_buffers();
  create_images();
  create_samplers();

  allocator.allocate_and_bind();

  create_image_views();
  stage_input_storage_buffer_data();
  stage_image_data();
  copy_uniform_buffer_data();

  create_descriptor_set_layout();
  create_descriptor_pool();
  create_descriptor_set();
  update_descriptor_set();
  create_pipeline();

  create_command_buffer();
  record_command_buffer();
}

void RenderJob::process_scene() {
  std::size_t vertex_count = scene.triangle_indices.size() * 3;
  vertex_positions.reserve(vertex_count);
  packed_vertex_data.reserve(vertex_count);
  for (const auto& vertex_indices : scene.triangle_indices) {
    for (auto i : vertex_indices) {
      const auto& vertex = scene.vertices[i];

      vertex_positions.push_back(
        glm::vec4(vertex.position, 0.0f)
      );

      PackedVertexData vertex_data {
        .normal_and_texcoord_u = glm::vec4(vertex.normal, vertex.texcoord.x),
        .tangent_and_texcoord_v = glm::vec4(vertex.tangent, vertex.texcoord.y),
        .bitangent = vertex.bitangnet,
        .material_index = vertex.material_index
      };
      packed_vertex_data.push_back(vertex_data);
    }
  }

  packed_bvh_nodes.resize(bvh_nodes.size());
  std::transform(
    bvh_nodes.begin(), bvh_nodes.end(), packed_bvh_nodes.begin(),
    [] (const BVHNode& node) {
      PackedBVHNode optimized_node {
        .aabb_min       = node.aabb.box_min,
        .left_or_begin  = node.triangle_count > 0 ? node.begin_index : node.left_child_index,
        .aabb_max       = node.aabb.box_max,
        .triangle_count = node.triangle_count
      };

      return optimized_node;
    }
  );
}

auto RenderJob::create_buffer(const BufferCreateInfo& create_info) 
    -> std::unique_ptr<vk::raii::Buffer> {
  vk::BufferCreateInfo vk_create_info {
    .size = create_info.size,
    .usage = create_info.usage,
    .sharingMode = vk::SharingMode::eExclusive
  };

  auto buffer = std::make_unique<vk::raii::Buffer>(device, vk_create_info);
  allocator.add_resource(*buffer, create_info.memory_flags);

  return buffer;
}

void RenderJob::create_buffers() {
  using enum vk::BufferUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  // Input Storage Buffer
  input_storage_buffer_size = 0;
  for (auto buffer_info : input_storage_buffer_infos) {
    input_storage_buffer_size += buffer_info.size;
  }

  input_staging_buffer = create_buffer({
    .size = input_storage_buffer_size,
    .usage = eTransferSrc,
    .memory_flags = eHostVisible | eHostCoherent
  });

  input_storage_buffer = create_buffer({
    .size = input_storage_buffer_size,
    .usage = eStorageBuffer | eTransferDst,
    .memory_flags = eDeviceLocal
  });

  // Output Storage Buffer
  output_image_pixel_count = config.image_width * config.image_height;
  output_storage_buffer_size = output_image_pixel_count * NR_CHANNELS * sizeof(float);

  output_storage_buffer = create_buffer({
    .size = output_storage_buffer_size,
    .usage = eStorageBuffer | eTransferSrc,
    .memory_flags = eDeviceLocal
  });

  output_unstaging_buffer = create_buffer({
    .size = output_storage_buffer_size,
    .usage = eTransferDst,
    .memory_flags = eHostVisible | eHostCoherent | eHostCached
  });

  // Uniform Buffer
  uniform_buffer_size = 0;
  for (auto buffer_info : uniform_buffer_infos) {
    uniform_buffer_size += buffer_info.size;
  }

  uniform_buffer = create_buffer({
    .size = uniform_buffer_size,
    .usage = eUniformBuffer,
    .memory_flags = eDeviceLocal | eHostVisible | eHostCoherent
  });
  
  // Image Staging Buffer
  image_staging_buffer_size = 0;
  for (const auto& image : scene.images) {
    image_staging_buffer_size += image.data.size() * sizeof(image.data[0]);
  }

  image_staging_buffer = create_buffer({
    .size = image_staging_buffer_size,
    .usage = eTransferSrc,
    .memory_flags = eHostVisible | eHostCoherent
  });
}

void RenderJob::create_images() {
  image_count = scene.images.size();
  images.reserve(image_count);
  for (const auto& image : scene.images) {
    vk::ImageCreateInfo image_create_info {
      .imageType = vk::ImageType::e2D,
      .format = vk::Format::eR8G8B8A8Unorm,
      .extent = { .width = image.width, .height = image.height, .depth = 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      .sharingMode = vk::SharingMode::eExclusive,
      .initialLayout = vk::ImageLayout::eUndefined
    };
    
    images.emplace_back(device, image_create_info);
    allocator.add_resource(images.back(), vk::MemoryPropertyFlagBits::eDeviceLocal);
  }
}

void RenderJob::create_samplers() {
  combined_image_sampler_count = scene.textures.size();

  samplers.reserve(scene.samplers.size());
  for (const auto& sampler : scene.samplers) {
    vk::SamplerCreateInfo create_info {
      .magFilter = get_filter_type(sampler.mag_filter),
      .minFilter = get_filter_type(sampler.min_filter),
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = get_wrap_type(sampler.wrap_s),
      .addressModeV = get_wrap_type(sampler.wrap_t),
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0f,
      .anisotropyEnable = false,
      .maxAnisotropy = 0.0f,
      .compareEnable = false,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .borderColor = vk::BorderColor::eIntOpaqueBlack,
      .unnormalizedCoordinates = false,
    };

    samplers.emplace_back(device, create_info);
  }
}

void RenderJob::create_image_views() {
  image_views.reserve(image_count);
  for (const auto& image : images) {
    vk::ImageViewCreateInfo image_view_create_info {
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = vk::Format::eR8G8B8A8Unorm,
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    
    image_views.emplace_back(device, image_view_create_info);
  }
}

void RenderJob::stage_input_storage_buffer_data() {
  auto bind_info = allocator.get_bind_info(*input_staging_buffer).value();
  void* dst_ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, input_storage_buffer_size);
  std::size_t offset = 0;
  for (auto [src_ptr, size] : input_storage_buffer_infos) {
    std::memcpy(static_cast<std::byte*>(dst_ptr) + offset, src_ptr, size);
    offset += size;
  }
  (*device).unmapMemory(bind_info.memory);
}

void RenderJob::copy_uniform_buffer_data() {
  auto bind_info = allocator.get_bind_info(*uniform_buffer).value();
  void* dst_ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, uniform_buffer_size);
  std::uint32_t offset = 0;
  for (auto [src_ptr, size] : uniform_buffer_infos) {
    std::memcpy(static_cast<std::byte*>(dst_ptr) + offset, src_ptr, size);
    offset += size;
  }
  (*device).unmapMemory(bind_info.memory);
}

void RenderJob::stage_image_data() {
  auto bind_info = allocator.get_bind_info(*image_staging_buffer).value();
  void* dst_ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, image_staging_buffer_size);
  std::size_t offset = 0;
  for (const auto& image : scene.images) {
    std::size_t image_size = image.data.size() * sizeof(image.data[0]);
    std::memcpy(static_cast<std::byte*>(dst_ptr) + offset, image.data.data(), image_size);
    offset += image_size;
  }
  (*device).unmapMemory(bind_info.memory);
}

void RenderJob::create_descriptor_set_layout() {
  std::uint32_t binding_index = 0;

  std::array<vk::DescriptorSetLayoutBinding, descriptor_count> layout_bindings{};
  for (std::uint32_t i = 0; i < storage_buffer_count; ++i) {
    layout_bindings[binding_index] = vk::DescriptorSetLayoutBinding {
      .binding = binding_index,
      .descriptorType = vk::DescriptorType::eStorageBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eCompute
    };

    ++binding_index;
  }

  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    layout_bindings[binding_index] = vk::DescriptorSetLayoutBinding {
      .binding = binding_index,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eCompute
    };

    ++binding_index;
  }

  layout_bindings[binding_index] = 
    vk::DescriptorSetLayoutBinding {
      .binding = binding_index,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = combined_image_sampler_count,
      .stageFlags = vk::ShaderStageFlagBits::eCompute
  };

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .bindingCount = static_cast<std::uint32_t>(layout_bindings.size()),
    .pBindings = layout_bindings.data()
  };

  descriptor_set_layout = 
    std::make_unique<vk::raii::DescriptorSetLayout>(device, descriptor_set_layout_create_info);
}

void RenderJob::create_descriptor_pool() {
  std::array<vk::DescriptorPoolSize, 3> pool_sizes{};

  pool_sizes[0] = vk::DescriptorPoolSize {
    .type = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = storage_buffer_count
  };

  pool_sizes[1] = vk::DescriptorPoolSize {
    .type = vk::DescriptorType::eUniformBuffer,
    .descriptorCount = uniform_buffer_count
  };

  pool_sizes[2] = vk::DescriptorPoolSize {
    .type = vk::DescriptorType::eCombinedImageSampler,
    .descriptorCount = combined_image_sampler_count
  };

  vk::DescriptorPoolCreateInfo pool_create_info {
    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    .maxSets = 1,
    .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
    .pPoolSizes = pool_sizes.data()
  };

  descriptor_pool = std::make_unique<vk::raii::DescriptorPool>(device, pool_create_info);
}

void RenderJob::create_descriptor_set() {
  vk::DescriptorSetAllocateInfo allocate_info {
    .descriptorPool = *descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &(**descriptor_set_layout)
  };

  vk::raii::DescriptorSets descriptor_sets { device, allocate_info };
  descriptor_set = std::make_unique<vk::raii::DescriptorSet>(std::move(descriptor_sets.front()));
}

void RenderJob::update_descriptor_set() {
  std::array<vk::DescriptorBufferInfo, storage_buffer_count> descriptor_storage_buffer_infos{};
  descriptor_storage_buffer_infos[0] = vk::DescriptorBufferInfo {
    .buffer = **output_storage_buffer,
    .offset = 0,
    .range  = output_storage_buffer_size
  };

  std::size_t input_storage_buffer_offset = 0;
  for (std::uint32_t i = 1; i < storage_buffer_count; ++i) {
    descriptor_storage_buffer_infos[i] = vk::DescriptorBufferInfo {
      .buffer = **input_storage_buffer,
      .offset = input_storage_buffer_offset,
      .range  = input_storage_buffer_infos[i - 1].size
    };
    input_storage_buffer_offset += input_storage_buffer_infos[i - 1].size;
  }

  std::array<vk::DescriptorBufferInfo, uniform_buffer_count> descriptor_uniform_buffer_infos{};
  std::size_t uniform_buffer_offset = 0;
  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    descriptor_uniform_buffer_infos[i] = vk::DescriptorBufferInfo {
      .buffer = **uniform_buffer,
      .offset = uniform_buffer_offset,
      .range = uniform_buffer_infos[i].size
    };
    uniform_buffer_offset += uniform_buffer_infos[i].size;
  }

  std::vector<vk::DescriptorImageInfo> descriptor_image_infos;
  descriptor_image_infos.reserve(combined_image_sampler_count);
  for (auto [sampler_index, image_index] : scene.textures) {
    vk::DescriptorImageInfo descriptor_image_info {
      .sampler = samplers[sampler_index],
      .imageView = image_views[image_index],
      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    descriptor_image_infos.push_back(descriptor_image_info);
  }

  std::array<vk::WriteDescriptorSet, descriptor_count> descriptor_writes{};
  std::uint32_t descriptor_binding_index = 0;

  for (std::uint32_t i = 0; i < storage_buffer_count; ++i) {
    descriptor_writes[descriptor_binding_index] = {
      .dstSet = *descriptor_set,
      .dstBinding = descriptor_binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = vk::DescriptorType::eStorageBuffer,
      .pBufferInfo = &descriptor_storage_buffer_infos[i]
    };

    ++descriptor_binding_index;
  }

  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    descriptor_writes[descriptor_binding_index] = {
      .dstSet = *descriptor_set,
      .dstBinding = descriptor_binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .pBufferInfo = &descriptor_uniform_buffer_infos[i]
    };

    ++descriptor_binding_index;
  }

  descriptor_writes[descriptor_binding_index] = vk::WriteDescriptorSet {
    .dstSet = *descriptor_set,
    .dstBinding = descriptor_binding_index,
    .dstArrayElement = 0,
    .descriptorCount = combined_image_sampler_count,
    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
    .pImageInfo = descriptor_image_infos.data()
  };

  device.updateDescriptorSets(descriptor_writes, {});
}

void RenderJob::create_pipeline() {
  auto compute_shader_code = read_binary_file("shaders/main.comp.spv");
  vk::ShaderModuleCreateInfo shader_module_create_info {
    .codeSize = compute_shader_code.size(),
    .pCode = reinterpret_cast<const std::uint32_t*>(compute_shader_code.data())
  };

  vk::raii::ShaderModule shader_module { device, shader_module_create_info };

  specialization_constants[0] = local_size_x;
  specialization_constants[1] = local_size_y;
  specialization_constants[2] = static_cast<std::uint32_t>(scene.materials.size());
  specialization_constants[3] = combined_image_sampler_count;
  
  std::array<vk::SpecializationMapEntry, specialization_constant_count> specialization_map_entries{};
  for (std::uint32_t i = 0; i < specialization_constant_count; ++i) {
    specialization_map_entries[i] = vk::SpecializationMapEntry {
      .constantID = i,
      .offset = static_cast<std::uint32_t>(i * sizeof(SpecializationConstant_t)),
      .size = sizeof(SpecializationConstant_t)
    };
  }

  vk::SpecializationInfo specialization_info {
    .mapEntryCount = static_cast<std::uint32_t>(specialization_map_entries.size()),
    .pMapEntries = specialization_map_entries.data(),
    .dataSize = sizeof(SpecializationConstant_t) * specialization_constant_count,
    .pData = &specialization_constants
  };

  vk::PipelineShaderStageCreateInfo shader_stage_create_info {
    .stage = vk::ShaderStageFlagBits::eCompute,
    .module = shader_module,
    .pName = "main",
    .pSpecializationInfo = &specialization_info
  };

  vk::PushConstantRange push_constant_range {
    .stageFlags = vk::ShaderStageFlagBits::eCompute,
    .offset = 0,
    .size = sizeof(PushConstants)
  };

  vk::PipelineLayoutCreateInfo layout_create_info {
    .setLayoutCount = 1,
    .pSetLayouts = &(**descriptor_set_layout),
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_constant_range
  };

  pipeline_layout = std::make_unique<vk::raii::PipelineLayout>(device, layout_create_info);

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
  command_buffer->reset();

  vk::CommandBufferBeginInfo begin_info {
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  };
  command_buffer->begin(begin_info);

  transition_images_for_copy();
  copy_input_resources();
  dispatch();
  copy_output_resources();

  command_buffer->end();
}

void RenderJob::transition_images_for_copy() {
  std::vector<vk::ImageMemoryBarrier2> barriers(image_count);
  for (std::uint32_t i = 0; i < image_count; ++i) {
    barriers[i] = vk::ImageMemoryBarrier2 {
      .srcStageMask = vk::PipelineStageFlagBits2::eNone,
      .srcAccessMask = vk::AccessFlagBits2::eNone,
      .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
      .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eTransferDstOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = images[i],
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
  }

  vk::DependencyInfo dependency_info {
    .imageMemoryBarrierCount = image_count,
    .pImageMemoryBarriers = barriers.data()
  };

  command_buffer->pipelineBarrier2(dependency_info);
}

void RenderJob::copy_input_resources() {
  vk::BufferCopy buffer_copy_info {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = input_storage_buffer_size
  };

  command_buffer->copyBuffer(
    *input_staging_buffer, *input_storage_buffer, { buffer_copy_info }
  );

  vk::BufferMemoryBarrier2 buffer_barrier {
    .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *input_storage_buffer,
    .offset = 0,
    .size = input_storage_buffer_size
  };

  std::vector<vk::ImageMemoryBarrier2> image_barriers(image_count);
  std::size_t offset = 0;
  for (std::uint32_t i = 0; i < image_count; ++i) {
    vk::BufferImageCopy buffer_image_copy_info {
      .bufferOffset = offset,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .imageOffset = { 0, 0, 0 },
      .imageExtent = { scene.images[i].width, scene.images[i].height, 1 }
    };

    offset += scene.images[i].data.size() * sizeof(scene.images[i].data[0]);

    command_buffer->copyBufferToImage(
      *image_staging_buffer, images[i], 
      vk::ImageLayout::eTransferDstOptimal, buffer_image_copy_info
    );

    image_barriers[i] = vk::ImageMemoryBarrier2 {
      .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = images[i],
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
  }

  vk::DependencyInfo dependency_info {
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &buffer_barrier,
    .imageMemoryBarrierCount = image_count,
    .pImageMemoryBarriers = image_barriers.data()
  };

  command_buffer->pipelineBarrier2(dependency_info);
}

void RenderJob::dispatch() {
  command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);

  command_buffer->bindDescriptorSets(
    vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, { *descriptor_set }, {}
  );

  PushConstants push_constants { config };
  command_buffer->pushConstants<PushConstants>(
    *pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, { push_constants }
  );

  std::uint32_t global_size_x = (config.image_height + local_size_x - 1) / local_size_x;
  std::uint32_t global_size_y = (config.image_width  + local_size_y - 1) / local_size_y;

  command_buffer->dispatch(global_size_x, global_size_y, 1);
}

void RenderJob::copy_output_resources() {
  vk::BufferMemoryBarrier2 memory_barrier {
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
    .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *output_storage_buffer,
    .offset = 0,
    .size = output_storage_buffer_size
  };
  
  vk::DependencyInfo dependency_info {
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &memory_barrier
  };

  command_buffer->pipelineBarrier2(dependency_info);

  vk::BufferCopy buffer_copy_info {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = output_storage_buffer_size
  };

  command_buffer->copyBuffer(
    *output_storage_buffer, *output_unstaging_buffer, { buffer_copy_info }
  );
}

auto RenderJob::render() const -> std::pair<const float*, std::size_t> {
  vk::SubmitInfo submit_info {
    .commandBufferCount = 1,
    .pCommandBuffers = &(**command_buffer),
  };

  vk::FenceCreateInfo fence_create_info{};
  vk::raii::Fence fence { device, fence_create_info };
  
  renderer.compute_queue->submit(submit_info, fence);
  while (device.waitForFences({ fence }, true, UINT32_MAX) != vk::Result::eSuccess) {};

  auto bind_info = allocator.get_bind_info(*output_unstaging_buffer).value();
  void* ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, output_storage_buffer_size);
  return std::make_pair(static_cast<const float*>(ptr), output_image_pixel_count * NR_CHANNELS);
}

PushConstants::PushConstants(const RenderConfig& config) :
    image_width { config.image_width }, image_height { config.image_height },
    seed { config.seed }, sample_count { config.sample_count },
    bg_color { config.bg_color } {

  float image_width = static_cast<float>(config.image_width);
  float image_height = static_cast<float>(config.image_height);
  const auto& camera = config.camera;

  glm::vec3 w = glm::normalize(camera.position - camera.lookat);
  glm::vec3 u = glm::normalize(glm::cross(camera.up, w));
  glm::vec3 v = glm::cross(w, u);

  float focal_length = glm::length(camera.position - camera.lookat);
  float theta = glm::radians(camera.vertical_fov);
  float viewport_height = 2.0f * glm::tan(theta / 2.0f) * focal_length;
  float viewport_width = viewport_height * image_width / image_height;

  glm::vec3 viewport_u = viewport_width * u;
  glm::vec3 viewport_v = viewport_height * -v;
  glm::vec3 pixel_delta_u = viewport_u / image_width;
  glm::vec3 pixel_delta_v = viewport_v / image_height;
  glm::vec3 viewport_upper_left = camera.position - focal_length * w - 0.5f * (viewport_u + viewport_v);
  glm::vec3 corner_pixel_pos = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

  this->camera = {
    .position = camera.position,
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

auto get_filter_type (Scene::SamplerFilter_t filter) -> vk::Filter {
  switch (filter) {
    case Scene::SamplerFilter_t::linear:
      return vk::Filter::eLinear;

    case Scene::SamplerFilter_t::nearest:
      return vk::Filter::eNearest;
  }
};

auto get_wrap_type(Scene::SamplerWarp_t wrap) -> vk::SamplerAddressMode {
  switch (wrap) {
    case Scene::SamplerWarp_t::repeat:
      return vk::SamplerAddressMode::eRepeat;

    case Scene::SamplerWarp_t::mirrored_repeat:
      return vk::SamplerAddressMode::eMirroredRepeat;

    case Scene::SamplerWarp_t::clamp_to_edge:
      return vk::SamplerAddressMode::eClampToEdge;
  }
};