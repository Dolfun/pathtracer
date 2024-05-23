#pragma once
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <atomic>
#include <unordered_map>

template<typename T>
concept ValidResource 
  = std::is_same_v<T, vk::raii::Buffer> || std::is_same_v<T, vk::raii::Image>;

class VkAllocator {
public:
  VkAllocator(const vk::raii::Device&, const vk::PhysicalDeviceMemoryProperties&);

  template <ValidResource T>
  void add_resource(const T& resource, vk::MemoryPropertyFlags flags);

  void allocate_and_bind();

private:
  uint32_t get_memory_type_index(uint32_t, vk::MemoryPropertyFlags);

  template <ValidResource T>
  struct ResourceMemoryRegionInfo {
    const T& resource;
    std::size_t offset;
  };

  struct MemoryTypeInfo {
    std::unique_ptr<vk::raii::DeviceMemory> memory;
    std::size_t memory_offset;
    std::vector<vk::BindBufferMemoryInfo> buffer_memory_regions;
    std::vector<vk::BindImageMemoryInfo> image_memory_regions;
  };

  const vk::raii::Device& device;
  vk::PhysicalDeviceMemoryProperties memory_properties;
  std::unordered_map<uint32_t, MemoryTypeInfo> memory_type_infos;
  std::atomic<bool> allocated;
};

template <ValidResource T>
void VkAllocator::add_resource(const T& resource, vk::MemoryPropertyFlags flags) {
  auto requirements = resource.getMemoryRequirements();
  uint32_t index = get_memory_type_index(requirements.memoryTypeBits, flags);
  auto& memory_type_info = memory_type_infos[index];

  std::size_t curr_offset = memory_type_info.memory_offset;
  std::size_t alignment = requirements.alignment;
  std::size_t alignment_correction = (alignment - curr_offset % alignment) % alignment;
  curr_offset += alignment_correction;

  memory_type_info.memory_offset = curr_offset + requirements.size;

  if constexpr (std::is_same_v<T, vk::raii::Buffer>) {
    vk::BindBufferMemoryInfo bind_info {
      .buffer = resource,
      .memoryOffset = curr_offset
    };
    memory_type_info.buffer_memory_regions.emplace_back(bind_info);
    
  } else {
    vk::BindImageMemoryInfo bind_info {
      .image = resource,
      .memoryOffset = curr_offset
    };
    memory_type_info.image_memory_regions.emplace_back(bind_info);
  }
}