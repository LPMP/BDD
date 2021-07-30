project(LPMP-BDD)

cmake_minimum_required(VERSION 2.8.12)

add_library(LPMP-BDD INTERFACE)
target_include_directories(LPMP-BDD INTERFACE include/)
target_compile_features(LPMP-BDD INTERFACE cxx_std_17)
target_compile_options(LPMP-BDD INTERFACE -march=native)
target_compile_options(LPMP-BDD INTERFACE -fPIC)
target_include_directories(LPMP-BDD INTERFACE external/Eigen)
target_include_directories(LPMP-BDD INTERFACE external/tsl-robin-map/include)
target_include_directories(LPMP-BDD INTERFACE external/PEGTL/include)
target_include_directories(LPMP-BDD INTERFACE external/CLI11/include)
target_include_directories(LPMP-BDD INTERFACE external/pybind11)

find_package(OpenMP REQUIRED)
target_link_libraries(LPMP-BDD INTERFACE ${OpenMP_CXX_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)
