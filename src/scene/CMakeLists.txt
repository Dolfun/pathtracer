add_library(scene scene.h scene.cc)
target_link_libraries(
  scene
  PUBLIC glm::glm
)
target_compile_features(scene PUBLIC cxx_std_23)
target_include_directories(scene PUBLIC .)