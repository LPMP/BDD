project (LPMP-BDD)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_POSITION_INDEPENDETN_CODE ON)

add_library(LPMP-BDD INTERFACE)
target_include_directories(LPMP-BDD INTERFACE include/)
target_compile_features(LPMP-BDD INTERFACE cxx_std_17)
target_compile_options(LPMP-BDD INTERFACE -march=native)
target_include_directories(LPMP-BDD INTERFACE external/LBDD/include)
target_include_directories(LPMP-BDD INTERFACE external/Eigen)
target_include_directories(LPMP-BDD INTERFACE external/tsl-robin-map/include)
target_include_directories(LPMP-BDD INTERFACE external/PEGTL/include)
target_include_directories(LPMP-BDD INTERFACE external/CLI11/include)

add_subdirectory(external/LBDD)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)
