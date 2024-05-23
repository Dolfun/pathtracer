#include "vk_allocator.h"

VkAllocator::VkAllocator(const VkManager& vk_manager)
  : device { vk_manager.get_device() },
  memory_properties { vk_manager.get_physical_device().getMemoryProperties() },
  allocated { false } {}

void VkAllocator::allocate_and_bind() {
  if (allocated) {
    throw std::runtime_error("allocate_and_bind called more than once.");
  }

  allocated = true;

  for (auto& [index, info] : memory_type_infos) {
    auto& [memory, offset, buffer_memory_regions, image_memory_regions] = info;

    vk::MemoryAllocateInfo allocate_info {
      .allocationSize = static_cast<vk::DeviceSize>(offset),
      .memoryTypeIndex = index
    };
    memory = std::make_unique<vk::raii::DeviceMemory>(device, allocate_info);

    auto set_memory_field = [&memory] (auto& bind_infos) {
      for (auto& bind_info : bind_infos) {
        bind_info.memory = *memory;
      }
    };

    set_memory_field(buffer_memory_regions);
    device.bindBufferMemory2(buffer_memory_regions);

    set_memory_field(image_memory_regions);
    device.bindImageMemory2(image_memory_regions);
  }
}

uint32_t VkAllocator::get_memory_type_index(uint32_t type_filter, vk::MemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((type_filter & (1 << i)) && 
        (memory_properties.memoryTypes[i].propertyFlags & flags) == flags) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type.");
}