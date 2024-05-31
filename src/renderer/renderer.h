#pragma once
#include <memory>
#include <optional>
#include <glm/vec3.hpp>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#if !defined(NDEBUG)
#define ENABLE_VALIDATION_LAYERS
#endif

#define NR_CHANNELS 4

struct RenderConfig {
  std::uint32_t image_width, image_height;
  std::uint32_t seed;
  std::uint32_t nr_samples;

  struct Camera {
    glm::vec3 center;
    glm::vec3 lookat;
    glm::vec3 up;
    float vertical_fov;
  } camera;
};

class Renderer {
public:
  Renderer();

  auto render(const RenderConfig&) const -> std::pair<const float*, std::size_t>;

  friend class RenderJob;

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