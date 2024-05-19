add_subdirectory(renderer)

add_executable(main main.cc)
target_link_libraries(main PRIVATE renderer)