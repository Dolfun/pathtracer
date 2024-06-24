#pragma once
#include <memory>
#include <optional>
#include "../scene.h"

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#if !defined(NDEBUG)
#define ENABLE_VALIDATION_LAYERS
#endif

#define NR_CHANNELS 4

struct RenderConfig {
  std::uint32_t resolution_x, resolution_y;
  std::uint32_t sample_count;
  glm::vec3 bg_color;
};

class Renderer {
public:
  Renderer(const std::uint32_t device_index = -1);

  auto render(Scene&, const RenderConfig&) const 
    -> std::pair<const unsigned char*, std::size_t>;

  friend class RenderJob;

private:
  void create_instance();
  void select_physical_device(const std::uint32_t);
  void select_queue_family_indices();
  void check_extensions_support() const;
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

void list_devices();