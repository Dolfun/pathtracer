set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(stb)
add_subdirectory(gltf_loader)
add_subdirectory(bvh)
add_subdirectory(renderer)

add_executable(main main.cc)
target_link_libraries(
  main
  PRIVATE
  stb
  renderer
  gltf_loader
  nlohmann_json::nlohmann_json
)