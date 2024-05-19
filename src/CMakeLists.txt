add_executable(main main.cc)
target_compile_features(main PRIVATE cxx_std_23)
target_link_libraries(main PRIVATE fmt::fmt)