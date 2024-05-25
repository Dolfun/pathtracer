#include "vk_allocator.h"

VkAllocator::VkAllocator(const vk::raii::Device& _device, const vk::PhysicalDeviceMemoryProperties& properties)
  : device { _device }, memory_properties { properties }, allocated { false } {}

void VkAllocator::allocate_and_bind() {
  if (allocated) {
    throw std::runtime_error("allocate_and_bind called more than once.");
  }

  allocated = true;

  for (auto& [index, info] : memory_type_infos) {
    auto& [memory, offset, buffer_memory_bind_infos, image_memory_bind_infos] = info;

    vk::MemoryAllocateInfo allocate_info {
      .allocationSize = static_cast<vk::DeviceSize>(offset),
      .memoryTypeIndex = index
    };
    memory = std::make_unique<vk::raii::DeviceMemory>(device, allocate_info);

    if (!buffer_memory_bind_infos.empty()) {
      for (auto& bind_info : buffer_memory_bind_infos) {
        bind_info.memory = *memory;
        buffer_memory_bind_info_map[bind_info.buffer] = bind_info;
      }
      device.bindBufferMemory2(buffer_memory_bind_infos);
    }
    
    if (!image_memory_bind_infos.empty()) {
      for (auto& bind_info : image_memory_bind_infos) {
        bind_info.memory = *memory;
        image_memory_bind_info_map[bind_info.image] = bind_info;
      }
      device.bindImageMemory2(image_memory_bind_infos);
    }
  }
}

auto VkAllocator::get_bind_info(const vk::raii::Buffer& buffer) const 
    -> std::optional<vk::BindBufferMemoryInfo> {
  auto it = buffer_memory_bind_info_map.find(buffer);
  if (it != buffer_memory_bind_info_map.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

auto VkAllocator::get_bind_info(const vk::raii::Image& image) const 
    -> std::optional<vk::BindImageMemoryInfo> {
  auto it = image_memory_bind_info_map.find(image);
  if (it != image_memory_bind_info_map.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

std::uint32_t VkAllocator::get_memory_type_index(std::uint32_t type_filter, vk::MemoryPropertyFlags flags) {
  for (auto i = 0u; i < memory_properties.memoryTypeCount; ++i) {
    if ((type_filter & (1 << i)) && 
        (memory_properties.memoryTypes[i].propertyFlags & flags) == flags) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type.");
}