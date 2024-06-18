#include "render_job.h"
#include "renderer.h"
#include "../timeit.h"
#include <fstream>
#include <glm/glm.hpp>

constexpr vk::ImageSubresourceRange single_image_subresource_range = {
  .aspectMask = vk::ImageAspectFlagBits::eColor,
  .baseMipLevel = 0,
  .levelCount = 1,
  .baseArrayLayer = 0,
  .layerCount = 1
};

auto read_binary_file(const std::string& path) -> std::vector<std::byte>;
auto get_filter_type (Scene::SamplerFilter_t filter) -> vk::Filter;
auto get_wrap_type(Scene::SamplerWarp_t wrap) -> vk::SamplerAddressMode;

RenderJob::RenderJob(const Renderer& _renderer, const RenderConfig& _config, Scene& _scene)
  : renderer { _renderer }, device { *renderer.device }, config { _config },
    allocator { *renderer.device, renderer.physical_device->getMemoryProperties() },
    scene { _scene } {

  timeit("Building BVH", [&] {
    bvh_nodes = build_bvh(scene, 16, bvh_max_depth);
  });

  initialize_work_group_size();

  process_scene();

  set_scene_data_storage_buffer_info<0>(vertex_positions);
  set_scene_data_storage_buffer_info<1>(packed_vertex_data);
  set_scene_data_storage_buffer_info<2>(packed_bvh_nodes);
  
  set_uniform_buffer_info<0>(scene.materials);
  set_uniform_buffer_info<1>(scene.directional_lights);
  set_uniform_buffer_info<2>(scene.point_lights);
  
  create_buffers();
  create_images();
  create_samplers();

  allocator.allocate_and_bind();

  create_image_views();
  stage_scene_data_storage_buffer();
  fill_uniform_buffer();
  stage_images();

  create_descriptor_set_layout();
  create_descriptor_pool();
  create_descriptor_set();
  update_descriptor_set();
  create_pipeline();

  create_command_buffer();
  record_command_buffer();
}

void RenderJob::initialize_work_group_size() {
  if (config.resolution_x == config.resolution_y) {
    local_size_x = local_size_y = 32;

  } else if (config.resolution_x > config.resolution_y) {
    local_size_x = 32, local_size_y = 16;

  } else {
    local_size_x = 16, local_size_y = 32;

  }
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
        .position = glm::vec4(vertex.position, 0.0f),
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
  create_scene_data_storage_buffer();
  create_uniform_buffer();
  create_image_staging_buffer();
  create_result_unstaging_buffer();
}

void RenderJob::create_scene_data_storage_buffer() {
  scene_data_storage_buffer_size = 0;
  for (auto buffer_info : scene_data_storage_buffer_infos) {
    scene_data_storage_buffer_size += buffer_info.size;
  }

  using enum vk::BufferUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  scene_data_staging_buffer = create_buffer({
    .size = scene_data_storage_buffer_size,
    .usage = eTransferSrc,
    .memory_flags = eHostVisible | eHostCoherent
  });

  scene_data_storage_buffer = create_buffer({
    .size = scene_data_storage_buffer_size,
    .usage = eStorageBuffer | eTransferDst,
    .memory_flags = eDeviceLocal
  });
}

void RenderJob::create_uniform_buffer() {
  uniform_buffer_size = 0;
  for (auto buffer_info : uniform_buffer_infos) {
    uniform_buffer_size += buffer_info.size;
  }

  using enum vk::BufferUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  uniform_buffer = create_buffer({
    .size = uniform_buffer_size,
    .usage = eUniformBuffer,
    .memory_flags = eDeviceLocal | eHostVisible | eHostCoherent
  });
}

void RenderJob::create_image_staging_buffer() {
  image_staging_buffer_size = 0;
  for (const auto& image : scene.images) {
    image_staging_buffer_size += image.data.size() * sizeof(image.data[0]);
  }

  using enum vk::BufferUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  image_staging_buffer = create_buffer({
    .size = image_staging_buffer_size,
    .usage = eTransferSrc,
    .memory_flags = eHostVisible | eHostCoherent
  });
}

void RenderJob::create_result_unstaging_buffer() {
  result_image_pixel_count = config.resolution_x * config.resolution_y;
  result_unstaging_buffer_size = result_image_pixel_count * NR_CHANNELS * sizeof(float);

  using enum vk::BufferUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  result_unstaging_buffer = create_buffer({
    .size = result_unstaging_buffer_size,
    .usage = eTransferDst,
    .memory_flags = eHostVisible | eHostCoherent | eHostCached
  });
}

auto RenderJob::create_image(const ImageCreateInfo& create_info)
    -> vk::raii::Image {
  
  vk::ImageCreateInfo vk_create_info {
    .imageType = vk::ImageType::e2D,
    .format = create_info.format,
    .extent = { 
      .width = create_info.width, 
      .height = create_info.height, 
      .depth = 1 
    },
    .mipLevels = 1,
    .arrayLayers = 1,
    .samples = vk::SampleCountFlagBits::e1,
    .tiling = vk::ImageTiling::eOptimal,
    .usage = create_info.usage,
    .sharingMode = vk::SharingMode::eExclusive,
    .initialLayout = vk::ImageLayout::eUndefined
  };
  
  vk::raii::Image image { device, vk_create_info };
  allocator.add_resource(image, create_info.memory_flags);

  return image;
}

void RenderJob::create_images() {
  using enum vk::Format;
  using enum vk::ImageUsageFlagBits;
  using enum vk::MemoryPropertyFlagBits;

  image_count = static_cast<std::uint32_t>(scene.images.size());
  images.reserve(image_count);
  for (const auto& image : scene.images) {
    ImageCreateInfo create_info {
      .format = eR8G8B8A8Unorm,
      .width = image.width,
      .height = image.height,
      .usage = eSampled | eTransferDst,
      .memory_flags = eDeviceLocal
    };

    images.emplace_back(create_image(create_info));
  }

  ImageCreateInfo create_info {
    .format = eR8G8B8A8Unorm,
    .width = config.resolution_x,
    .height = config.resolution_y,
    .usage = eStorage | eTransferSrc,
    .memory_flags = eDeviceLocal
  };

  result_storage_image = std::make_unique<vk::raii::Image>(create_image(create_info));
}

void RenderJob::create_samplers() {
  combined_image_sampler_count = static_cast<std::uint32_t>(scene.textures.size());

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

auto RenderJob::create_image_view(const vk::raii::Image& image, vk::Format format)
    -> vk::raii::ImageView {
    
  vk::ImageViewCreateInfo create_info {
    .image = image,
    .viewType = vk::ImageViewType::e2D,
    .format = format,
    .subresourceRange = single_image_subresource_range
  };

  return vk::raii::ImageView { device, create_info };
}

void RenderJob::create_image_views() {
  using enum vk::Format;

  image_views.reserve(image_count);
  for (const auto& image : images) {
    image_views.emplace_back(create_image_view(image, eR8G8B8A8Unorm));
  }

  result_storage_image_view = std::make_unique<vk::raii::ImageView>(
    create_image_view(*result_storage_image, eR8G8B8A8Unorm)
  );
}

void RenderJob::stage_scene_data_storage_buffer() {
  auto bind_info = allocator.get_bind_info(*scene_data_staging_buffer).value();
  void* dst_ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, scene_data_storage_buffer_size);
  std::size_t offset = 0;
  for (auto [src_ptr, size] : scene_data_storage_buffer_infos) {
    std::memcpy(static_cast<std::byte*>(dst_ptr) + offset, src_ptr, size);
    offset += size;
  }
  (*device).unmapMemory(bind_info.memory);
}

void RenderJob::fill_uniform_buffer() {
  auto bind_info = allocator.get_bind_info(*uniform_buffer).value();
  void* dst_ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, uniform_buffer_size);
  std::uint32_t offset = 0;
  for (auto [src_ptr, size] : uniform_buffer_infos) {
    std::memcpy(static_cast<std::byte*>(dst_ptr) + offset, src_ptr, size);
    offset += static_cast<std::uint32_t>(size);
  }
  (*device).unmapMemory(bind_info.memory);
}

void RenderJob::stage_images() {
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

  using enum vk::DescriptorType;
  using enum vk::ShaderStageFlagBits;

  std::array<vk::DescriptorSetLayoutBinding, descriptor_count> layout_bindings{};
  for (std::uint32_t i = 0; i < scene_data_storage_buffer_count; ++i) {
    layout_bindings[binding_index] = {
      .binding = binding_index,
      .descriptorType = eStorageBuffer,
      .descriptorCount = 1,
      .stageFlags = eCompute
    };

    ++binding_index;
  }

  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    layout_bindings[binding_index] = {
      .binding = binding_index,
      .descriptorType = eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags = eCompute
    };

    ++binding_index;
  }

  layout_bindings[binding_index] = {
    .binding = binding_index,
    .descriptorType = eCombinedImageSampler,
    .descriptorCount = combined_image_sampler_count,
    .stageFlags = eCompute
  };
  ++binding_index;

  layout_bindings[binding_index] = {
    .binding = binding_index,
    .descriptorType = eStorageImage,
    .descriptorCount = 1,
    .stageFlags = eCompute
  };
  ++binding_index;
  
  assert(binding_index == descriptor_count);

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .bindingCount = static_cast<std::uint32_t>(layout_bindings.size()),
    .pBindings = layout_bindings.data()
  };

  descriptor_set_layout = 
    std::make_unique<vk::raii::DescriptorSetLayout>(device, descriptor_set_layout_create_info);
}

void RenderJob::create_descriptor_pool() {
  std::array<vk::DescriptorPoolSize, 4> pool_sizes{};
  using enum vk::DescriptorType;

  pool_sizes[0] = {
    .type = eStorageBuffer,
    .descriptorCount = scene_data_storage_buffer_count
  };

  pool_sizes[1] = {
    .type = eUniformBuffer,
    .descriptorCount = uniform_buffer_count
  };

  pool_sizes[2] = {
    .type = eCombinedImageSampler,
    .descriptorCount = combined_image_sampler_count
  };

  pool_sizes[3] = {
    .type = eStorageImage,
    .descriptorCount = 1
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
  std::array<vk::WriteDescriptorSet, descriptor_count> descriptor_writes{};
  std::uint32_t binding_index = 0;
  using enum vk::DescriptorType;

  // Scene Data Storage Buffer
  std::array<vk::DescriptorBufferInfo, scene_data_storage_buffer_count> descriptor_storage_buffer_infos{};
  std::size_t scene_data_storage_buffer_offset = 0;
  for (std::uint32_t i = 0; i < scene_data_storage_buffer_count; ++i) {
    descriptor_storage_buffer_infos[i] = {
      .buffer = **scene_data_storage_buffer,
      .offset = scene_data_storage_buffer_offset,
      .range  = scene_data_storage_buffer_infos[i].size
    };
    scene_data_storage_buffer_offset += scene_data_storage_buffer_infos[i].size;
  }

  for (std::uint32_t i = 0; i < scene_data_storage_buffer_count; ++i) {
    descriptor_writes[binding_index] = {
      .dstSet = *descriptor_set,
      .dstBinding = binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = eStorageBuffer,
      .pBufferInfo = &descriptor_storage_buffer_infos[i]
    };

    ++binding_index;
  }

  // Uniform Buffer
  std::array<vk::DescriptorBufferInfo, uniform_buffer_count> descriptor_uniform_buffer_infos{};
  std::size_t uniform_buffer_offset = 0;
  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    descriptor_uniform_buffer_infos[i] = {
      .buffer = **uniform_buffer,
      .offset = uniform_buffer_offset,
      .range = uniform_buffer_infos[i].size
    };
    uniform_buffer_offset += uniform_buffer_infos[i].size;
  }

  for (std::uint32_t i = 0; i < uniform_buffer_count; ++i) {
    descriptor_writes[binding_index] = {
      .dstSet = *descriptor_set,
      .dstBinding = binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .pBufferInfo = &descriptor_uniform_buffer_infos[i]
    };

    ++binding_index;
  }

  // Combined Image Sampler
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

  descriptor_writes[binding_index] = {
    .dstSet = *descriptor_set,
    .dstBinding = binding_index,
    .dstArrayElement = 0,
    .descriptorCount = combined_image_sampler_count,
    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
    .pImageInfo = descriptor_image_infos.data()
  };
  ++binding_index;

  // Result Storage Image
  vk::DescriptorImageInfo descriptor_result_storage_image_info {
    .imageView = *result_storage_image_view,
    .imageLayout = vk::ImageLayout::eGeneral,
  };

  descriptor_writes[binding_index] = {
    .dstSet = *descriptor_set,
    .dstBinding = binding_index,
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType = vk::DescriptorType::eStorageImage,
    .pImageInfo = &descriptor_result_storage_image_info
  };
  ++binding_index;

  assert(binding_index == descriptor_count);

  device.updateDescriptorSets(descriptor_writes, {});
}

void RenderJob::create_pipeline() {
  auto compute_shader_code = read_binary_file("shaders/main.comp.spv");
  vk::ShaderModuleCreateInfo shader_module_create_info {
    .codeSize = compute_shader_code.size(),
    .pCode = reinterpret_cast<const std::uint32_t*>(compute_shader_code.data())
  };

  vk::raii::ShaderModule shader_module { device, shader_module_create_info };

  specialization_constants = {
    local_size_x,                                                    // 0
    local_size_y,                                                    // 1
    static_cast<std::uint32_t>(scene.materials.size()),              // 2
    static_cast<std::uint32_t>(scene.directional_lights.size() - 1), // 3
    static_cast<std::uint32_t>(scene.point_lights.size() - 1),       // 4
    bvh_max_depth,                                                   // 5
  };
  
  std::array<vk::SpecializationMapEntry, specialization_constant_count> specialization_map_entries{};
  for (std::uint32_t i = 0; i < specialization_constant_count; ++i) {
    specialization_map_entries[i] = {
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
  command_pool->reset();

  vk::CommandBufferBeginInfo begin_info {
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  };
  command_buffer->begin(begin_info);

  transition_images_for_usage();
  copy_scene_data_to_device();
  dispatch_compute_shader();
  copy_result_to_host();

  command_buffer->end();
}

void RenderJob::transition_images_for_usage() {
  std::vector<vk::ImageMemoryBarrier2> barriers(image_count + 1);
  for (std::uint32_t i = 0; i < image_count; ++i) {
    barriers[i] = {
      .srcStageMask  = vk::PipelineStageFlagBits2::eNone,
      .srcAccessMask = vk::AccessFlagBits2::eNone,
      .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
      .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eTransferDstOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = images[i],
      .subresourceRange = single_image_subresource_range
    };
  }

  barriers[image_count] = {
    .srcStageMask  = vk::PipelineStageFlagBits2::eNone,
    .srcAccessMask = vk::AccessFlagBits2::eNone,
    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .oldLayout = vk::ImageLayout::eUndefined,
    .newLayout = vk::ImageLayout::eGeneral,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .image = *result_storage_image,
    .subresourceRange = single_image_subresource_range
  };

  vk::DependencyInfo dependency_info {
    .imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
    .pImageMemoryBarriers = barriers.data()
  };

  command_buffer->pipelineBarrier2(dependency_info);
}

void RenderJob::copy_scene_data_to_device() {
  vk::BufferCopy buffer_copy_info {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = scene_data_storage_buffer_size
  };

  command_buffer->copyBuffer(
    *scene_data_staging_buffer, *scene_data_storage_buffer, { buffer_copy_info }
  );

  vk::BufferMemoryBarrier2 buffer_barrier {
    .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .buffer = *scene_data_storage_buffer,
    .offset = 0,
    .size = scene_data_storage_buffer_size
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

    image_barriers[i] = {
      .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = images[i],
      .subresourceRange = single_image_subresource_range
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

void RenderJob::dispatch_compute_shader() {
  command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);

  command_buffer->bindDescriptorSets(
    vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, { *descriptor_set }, {}
  );

  PushConstants push_constants { config, scene };
  command_buffer->pushConstants<PushConstants>(
    *pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, { push_constants }
  );

  std::uint32_t global_size_x = (config.resolution_x + local_size_x - 1) / local_size_x;
  std::uint32_t global_size_y = (config.resolution_y + local_size_y - 1) / local_size_y;

  command_buffer->dispatch(global_size_x, global_size_y, 1);
}

void RenderJob::copy_result_to_host() {
  vk::ImageMemoryBarrier2 image_barrier {
    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask  = vk::PipelineStageFlagBits2::eCopy,
    .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    .oldLayout = vk::ImageLayout::eGeneral,
    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
    .image = *result_storage_image,
    .subresourceRange = single_image_subresource_range
  };

  vk::DependencyInfo dependency_info {
    .imageMemoryBarrierCount = 1,
    .pImageMemoryBarriers = &image_barrier
  };

  command_buffer->pipelineBarrier2(dependency_info);

  vk::BufferImageCopy buffer_image_copy_info {
    .bufferOffset = 0,
    .bufferRowLength = 0,
    .bufferImageHeight = 0,
    .imageSubresource = {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .mipLevel = 0,
      .baseArrayLayer = 0,
      .layerCount = 1
    },
    .imageOffset = { 0, 0, 0 },
    .imageExtent = { config.resolution_x, config.resolution_y, 1 }
  };

  command_buffer->copyImageToBuffer(
    *result_storage_image, vk::ImageLayout::eTransferSrcOptimal, 
    *result_unstaging_buffer, { buffer_image_copy_info }
  );
}

auto RenderJob::render() const -> std::pair<const unsigned char*, std::size_t> {
  vk::SubmitInfo submit_info {
    .commandBufferCount = 1,
    .pCommandBuffers = &(**command_buffer),
  };

  vk::FenceCreateInfo fence_create_info{};
  vk::raii::Fence fence { device, fence_create_info };
  
  renderer.compute_queue->submit(submit_info, fence);
  while (device.waitForFences({ fence }, true, UINT32_MAX) != vk::Result::eSuccess) {};

  auto bind_info = allocator.get_bind_info(*result_unstaging_buffer).value();
  void* ptr = (*device).mapMemory(bind_info.memory, bind_info.memoryOffset, result_unstaging_buffer_size);
  return std::make_pair(static_cast<const unsigned char*>(ptr), result_image_pixel_count * NR_CHANNELS);
}

PushConstants::PushConstants(const RenderConfig& config, const Scene& scene) :
    resolution_x { config.resolution_x }, resolution_y { config.resolution_y },
    seed { config.seed }, sample_count { config.sample_count },
    bg_color { config.bg_color } {

  float resolution_x = static_cast<float>(config.resolution_x);
  float resolution_y = static_cast<float>(config.resolution_y);
  const auto& camera = scene.camera;

  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 w = glm::normalize(-camera.lookat);
  glm::vec3 u = glm::normalize(glm::cross(up, w));
  glm::vec3 v = glm::cross(w, u);

  float viewport_height = 2.0f;
  float viewport_width = viewport_height * resolution_x / resolution_y;
  float theta = camera.vertical_fov;
  float focal_length = 1.0f / glm::tan(theta / 2.0f);

  glm::vec3 viewport_u = viewport_width * u;
  glm::vec3 viewport_v = viewport_height * -v;
  glm::vec3 pixel_delta_u = viewport_u / resolution_x;
  glm::vec3 pixel_delta_v = viewport_v / resolution_y;
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

    default:
      return vk::Filter::eLinear;
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

    default:
      return vk::SamplerAddressMode::eRepeat;
  }
};