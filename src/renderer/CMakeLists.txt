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

find_program(GLSLC glslc REQUIRED)
set(SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(
  SHADER_SOURCES 
  main.comp
)

foreach(SHADER_SOURCE ${SHADER_SOURCES})
  set(SHADER_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_SOURCE}")
  set(SHADER_OUTPUT_PATH "${SHADER_OUTPUT_DIR}/${SHADER_SOURCE}.spv")

  add_custom_command(
    OUTPUT ${SHADER_OUTPUT_PATH}
    COMMAND ${GLSLC} ${SHADER_SOURCE_PATH} -o ${SHADER_OUTPUT_PATH}
    DEPENDS ${SHADER_SOURCE_PATH}
    VERBATIM
    COMMENT "Compiling ${SHADER_SOURCE}"
  )

  target_sources(renderer PRIVATE ${SHADER_OUTPUT_PATH})
endforeach()