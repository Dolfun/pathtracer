add_library(gltf_loader gltf_loader.h gltf_loader.cc)
target_link_libraries(
  gltf_loader
  PUBLIC  glm::glm
  PRIVATE fmt::fmt tinygltf
)
target_compile_features(gltf_loader PUBLIC cxx_std_23)
target_include_directories(gltf_loader PUBLIC .)