add_library(bvh bvh.h bvh.cc)
target_link_libraries(bvh PUBLIC glm::glm)
target_compile_features(bvh PUBLIC cxx_std_23)
target_include_directories(bvh PUBLIC .)