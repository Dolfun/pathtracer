add_subdirectory(fmt)
add_subdirectory(glm)

add_library(stb_headers INTERFACE)
target_include_directories(stb_headers INTERFACE stb)