#include "renderer.h"
#include "render_job.h"
#include "../timeit.h"
#include <fmt/core.h>
#include <fmt/color.h>

Renderer::Renderer(const std::uint32_t device_id) {
  create_instance();
  select_physical_device(device_id);
  select_queue_family_indices();
  check_extensions_support();
  create_logical_device();
}

void Renderer::create_instance() {
  vk::ApplicationInfo application_info {
    .apiVersion = VK_API_VERSION_1_3
  };

#ifdef ENABLE_VALIDATION_LAYERS
  using enum vk::DebugUtilsMessageSeverityFlagBitsEXT;
  using enum vk::DebugUtilsMessageTypeFlagBitsEXT;
  vk::DebugUtilsMessengerCreateInfoEXT debug_messenger_create_info = {
    .messageSeverity = eWarning | eError,
    .messageType = eGeneral | eValidation | ePerformance,
    .pfnUserCallback = debug_callback,
  };

  std::array layers = { "VK_LAYER_KHRONOS_validation" };
  std::array extensions = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

  vk::InstanceCreateInfo create_info {
    .pNext = &debug_messenger_create_info,
    .pApplicationInfo = &application_info,
    .enabledLayerCount = static_cast<std::uint32_t>(layers.size()),
    .ppEnabledLayerNames = layers.data(),
    .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data()
  };

  instance = std::make_unique<vk::raii::Instance>(context, create_info);
  debug_messenger = std::make_unique<vk::raii::DebugUtilsMessengerEXT>(
    *instance, debug_messenger_create_info
  );

#else
  vk::InstanceCreateInfo create_info {
    .pApplicationInfo = &application_info,
  };

  instance = std::make_unique<vk::raii::Instance>(context, create_info);
#endif
}

void Renderer::select_physical_device(const std::uint32_t device_id) {
  vk::raii::PhysicalDevices physical_devices { *instance };

  if (device_id > 0 && device_id < physical_devices.size()) {
    physical_device = std::make_unique<vk::raii::PhysicalDevice>(physical_devices[device_id]);
    return;
  }

  for (const auto& device : physical_devices) {
    if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      physical_device = std::make_unique<vk::raii::PhysicalDevice>(device);
      return;
    }
  }

  physical_device = std::make_unique<vk::raii::PhysicalDevice>(physical_devices.front());
}

void Renderer::select_queue_family_indices() {
  auto properties = physical_device->getQueueFamilyProperties();
  for (auto i = 0u; i < static_cast<std::uint32_t>(properties.size()); ++i) {
    if (properties[i].queueFlags & vk::QueueFlagBits::eCompute) {
      compute_family_index = i;

      if (!(properties[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
        break;
      }
    }
  }

  if (!compute_family_index.has_value()) {
    throw std::runtime_error("Failed to find a compute queue.");
  }
}

void Renderer::check_extensions_support() const {
  auto features = physical_device->getFeatures2<
    vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceSynchronization2Features,
    vk::PhysicalDeviceMaintenance4Features,
    vk::PhysicalDeviceDescriptorIndexingFeatures>();

  if (!features.get<vk::PhysicalDeviceSynchronization2Features>().synchronization2) {
    throw std::runtime_error("Synchronization2 is not supported!");
  }

  if (!features.get<vk::PhysicalDeviceMaintenance4Features>().maintenance4) {
    throw std::runtime_error("Maintenance4 is not supported!");
  }

  if (!features.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().shaderSampledImageArrayNonUniformIndexing) {
    throw std::runtime_error("shaderSampledImageArrayNonUniformIndexing is not supported!");
  }

  if (!features.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().runtimeDescriptorArray) {
    throw std::runtime_error("runtimeDescriptorArray is not supported!");
  }
}

void Renderer::create_logical_device() {
  float queue_priority = 1.0f;
  vk::DeviceQueueCreateInfo queue_create_info {
    .queueFamilyIndex = compute_family_index.value(),
    .queueCount = 1,
    .pQueuePriorities = &queue_priority,
  };

  vk::StructureChain<
    vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceSynchronization2Features,
    vk::PhysicalDeviceMaintenance4Features,
    vk::PhysicalDeviceDescriptorIndexingFeatures> features;

  features.get<vk::PhysicalDeviceSynchronization2Features>().synchronization2 = true;
  features.get<vk::PhysicalDeviceMaintenance4Features>().maintenance4 = true;
  features.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().shaderSampledImageArrayNonUniformIndexing = true;
  features.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().runtimeDescriptorArray = true;

  vk::DeviceCreateInfo device_create_info {
    .pNext = &features.get<vk::PhysicalDeviceFeatures2>(),
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_create_info,
  };

  device = std::make_unique<vk::raii::Device>(*physical_device, device_create_info);
  
  compute_queue = std::make_unique<vk::raii::Queue>(
    device->getQueue(compute_family_index.value(), 0)
  );
}

#ifdef ENABLE_VALIDATION_LAYERS
VKAPI_ATTR VkBool32 VKAPI_CALL Renderer::debug_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT severity,
  VkDebugUtilsMessageTypeFlagsEXT,
  const VkDebugUtilsMessengerCallbackDataEXT* data,
  void*) {

  fmt::color color { fmt::color::white };
  switch (severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      color = fmt::color::white;
      break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      color = fmt::color::green;
      break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      color = fmt::color::yellow;
      break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      color = fmt::color::red;
      break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
      color = fmt::color::dark_red;
      break;
  }

  std::string error = fmt::format("[[{}]] {}\n", data->pMessageIdName, data->pMessage);
  fmt::print(fmt::fg(color), "{}\n", error);

  return VK_FALSE;
}
#endif

auto Renderer::render(Scene& scene, const RenderConfig& config) const
    -> std::pair<const unsigned char*, std::size_t> {

  RenderJob render_job { *this, config, scene };

  std::pair<const unsigned char*, std::size_t> result;
  timeit("Rendering", [&] {
    result = render_job.render();
  });

  return result;
}

void list_devices() {
  vk::ApplicationInfo application_info {
    .apiVersion = VK_API_VERSION_1_3
  };

  vk::InstanceCreateInfo create_info {
    .pApplicationInfo = &application_info,
  };

  vk::raii::Context context;
  vk::raii::Instance instance { context, create_info };
  vk::raii::PhysicalDevices devices { instance };

  for (std::size_t i = 0; i < devices.size(); ++i) {
    auto name = devices[i].getProperties().deviceName;
    std::string name_str { name.begin(), name.end() };
    fmt::println("{}: {}", i, name_str);
  }
}