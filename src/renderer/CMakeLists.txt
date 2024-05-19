add_library(renderer renderer.h renderer.cc)
target_link_libraries(renderer PUBLIC fmt::fmt Vulkan::Vulkan)
target_compile_features(renderer PUBLIC cxx_std_23)
target_include_directories(renderer PUBLIC .)