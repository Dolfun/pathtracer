add_subdirectory(fmt)
add_subdirectory(glm)

add_library(stb_headers INTERFACE)
target_include_directories(stb_headers INTERFACE stb)

set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
add_subdirectory(tinygltf)