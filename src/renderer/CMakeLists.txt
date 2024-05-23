set (
  sources
  vk_manager.h
  vk_manager.cc
  vk_allocator.h
  vk_allocator.cc
  render_config.h
  render_job.h
  render_job.cc
  renderer.h
  renderer.cc
)
add_library(renderer ${sources})
target_link_libraries(renderer PUBLIC fmt::fmt Vulkan::Vulkan)
target_compile_features(renderer PUBLIC cxx_std_23)
target_include_directories(renderer PUBLIC .)