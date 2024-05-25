#pragma once
#include <memory>
#include <optional>
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#if !defined(NDEBUG)
#define ENABLE_VALIDATION_LAYERS
#endif

class VkManager {
public:
  VkManager();

  const auto& get_physical_device() const noexcept { return *physical_device; }
  const auto& get_device() const noexcept { return *device; }
  const auto& get_compute_queue() const noexcept { return *compute_queue; }
  std::uint32_t get_queue_family_index() const noexcept { return compute_family_index.value(); }

private:
  void create_instance();
  void select_physical_device();
  void select_queue_family_indices();
  void create_logical_device();

  vk::raii::Context context;
  std::unique_ptr<vk::raii::Instance> instance;
  std::unique_ptr<vk::raii::PhysicalDevice> physical_device;
  std::optional<std::uint32_t> compute_family_index;
  std::unique_ptr<vk::raii::Device> device;
  std::unique_ptr<vk::raii::Queue> compute_queue;

#ifdef ENABLE_VALIDATION_LAYERS
  std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debug_messenger;
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*,
    void*);
#endif
};