set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(stb)
add_subdirectory(gltf_loader)
add_subdirectory(renderer)

add_executable(main main.cc)
target_link_libraries(main PRIVATE renderer stb gltf_loader)