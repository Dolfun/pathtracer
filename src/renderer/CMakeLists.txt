set (
  sources
  renderer.h
  renderer.cc
  vulkan_manager.h
  vulkan_manager.cc
)
add_library(renderer ${sources})
target_link_libraries(renderer PUBLIC fmt::fmt Vulkan::Vulkan)
target_compile_features(renderer PUBLIC cxx_std_23)
target_include_directories(renderer PUBLIC .)