#include "vk_manager.h"
#include <fmt/core.h>
#include <fmt/color.h>

VkManager::VkManager() {
  create_instance();
  select_physical_device();
  select_queue_family_indices();
  create_logical_device();
}

void VkManager::create_instance() {
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
    .enabledLayerCount = static_cast<uint32_t>(layers.size()),
    .ppEnabledLayerNames = layers.data(),
    .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
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

void VkManager::select_physical_device() {
  vk::raii::PhysicalDevices physical_devices { *instance };

  for (const auto& device : physical_devices) {
    if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      physical_device = std::make_unique<vk::raii::PhysicalDevice>(device);
      return;
    }
  }

  physical_device = std::make_unique<vk::raii::PhysicalDevice>(physical_devices.front());
}

void VkManager::select_queue_family_indices() {
  auto properties = physical_device->getQueueFamilyProperties();
  for (uint32_t i = 0; i < static_cast<uint32_t>(properties.size()); ++i) {
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

void VkManager::create_logical_device() {
  float queue_priority = 1.0f;
  vk::DeviceQueueCreateInfo queue_create_info {
    .queueFamilyIndex = compute_family_index.value(),
    .queueCount = 1,
    .pQueuePriorities = &queue_priority,
  };

  vk::PhysicalDeviceSynchronization2Features sync2_features {
    .synchronization2 = true,
  };

  vk::DeviceCreateInfo device_create_info {
    .pNext = &sync2_features,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_create_info,
  };

  device = std::make_unique<vk::raii::Device>(*physical_device, device_create_info);
  
  compute_queue = std::make_unique<vk::raii::Queue>(
    device->getQueue(compute_family_index.value(), 0)
  );
}

#ifdef ENABLE_VALIDATION_LAYERS
VKAPI_ATTR VkBool32 VKAPI_CALL VkManager::debug_callback(
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
  fmt::print(fmt::fg(color), "{}", error);

  return VK_FALSE;
}
#endif