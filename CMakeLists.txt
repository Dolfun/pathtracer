cmake_minimum_required(VERSION 3.10)
project(pathtracer-vulkan)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

add_subdirectory(src)