set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(stb)
add_subdirectory(gltf_loader)
add_subdirectory(bvh)
add_subdirectory(renderer)

add_executable(pathtracer main.cc)
target_link_libraries(
  pathtracer
  PRIVATE
  stb
  renderer
  gltf_loader
  nlohmann_json::nlohmann_json
)