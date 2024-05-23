#include "renderer.h"
#include "render_job.h"
#include <fstream>

Renderer::Renderer() : device { vk_manager.get_device() } {
  create_descriptors();
  create_compute_pipeline();
}

auto Renderer::read_binary_file(const std::string& path) -> std::vector<std::byte> {
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

void Renderer::create_descriptors() {
  // Descriptor set layout
  std::array<vk::DescriptorSetLayoutBinding, 1> layout_bindings{};
  layout_bindings[0] = {
    .binding = 0,
    .descriptorType = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1,
    .stageFlags = vk::ShaderStageFlagBits::eCompute
  };

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
    .pBindings = layout_bindings.data()
  };
  descriptor_set_layout = 
    std::make_unique<vk::raii::DescriptorSetLayout>(device, descriptor_set_layout_create_info);

  // Descriptor pool
  vk::DescriptorPoolSize pool_size {
    .type = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1
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
}

void Renderer::create_compute_pipeline() {
  // Shader Stage
  auto compute_shader_code = read_binary_file("shaders/main.comp.spv");
  vk::ShaderModuleCreateInfo shader_module_create_info {
    .codeSize = compute_shader_code.size(),
    .pCode = reinterpret_cast<const uint32_t*>(compute_shader_code.data())
  };
  vk::raii::ShaderModule shader_module { device, shader_module_create_info };

  vk::PipelineShaderStageCreateInfo shader_stage_create_info {
    .stage = vk::ShaderStageFlagBits::eCompute,
    .module = shader_module,
    .pName = "main"
  };

  // Pipeline layout
  vk::PipelineLayoutCreateInfo layout_create_info {
    .setLayoutCount = 1,
    .pSetLayouts = &(**descriptor_set_layout)
  };
  pipeline_layout = 
    std::make_unique<vk::raii::PipelineLayout>(device, layout_create_info);

  // Pipeline
  vk::ComputePipelineCreateInfo create_info {
    .stage = shader_stage_create_info,
    .layout = *pipeline_layout
  };
  compute_pipeline = 
    std::make_unique<vk::raii::Pipeline>(device, nullptr, create_info);
}

auto Renderer::render(const RenderConfig& config) const -> std::vector<std::byte> {
  RenderJob render_job { *this, config };

  auto result = render_job.render();

  return result;
}