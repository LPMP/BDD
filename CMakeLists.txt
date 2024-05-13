project(LPMP-BDD
#    VERSION 0.2
    DESCRIPTION "Lagrange decomposition with BDDs for ILPs"
    HOMEPAGE_URL "https://github.com/LPMP/BDD"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_CUDA "Compile with CUDA support" OFF)
option(WITH_REGRESSION_TEST "Regression tests on additional downloaded instances" OFF)

if(WITH_CUDA)
    message(STATUS "Compiling with CUDA support")
    add_definitions(-DWITH_CUDA)
    enable_language(CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    message("Compiling without CUDA support")
endif()

cmake_minimum_required(VERSION 3.20) # does not work with 3.13

#set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES OFF)


add_library(LPMP-BDD INTERFACE)
target_include_directories(LPMP-BDD INTERFACE include/)
target_compile_features(LPMP-BDD INTERFACE cxx_std_17)
target_compile_options(LPMP-BDD INTERFACE -fPIC)
target_compile_options(LPMP-BDD INTERFACE -fvisibility=hidden)
target_compile_options(LPMP-BDD INTERFACE -fvisibility-inlines-hidden)

# external dependencies
set(CPM_DOWNLOAD_VERSION 0.39.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:taocpp/PEGTL#main")
target_include_directories(LPMP-BDD INTERFACE "${PEGTL_SOURCE_DIR}/include")

CPMAddPackage("gl:libeigen/eigen#master")
target_include_directories(LPMP-BDD INTERFACE "${eigen_SOURCE_DIR}")

CPMAddPackage("gh:Tessil/robin-map#master")
target_include_directories(LPMP-BDD INTERFACE "${robin-map_SOURCE_DIR}/include")

set(PYBIND11_FINDPYTHON ON) # fix for empty list problem
CPMAddPackage(
    NAME pybind11
    GIT_TAG v2.11
    GITHUB_REPOSITORY pybind/pybind11
    OPTIONS
    "PYBIND11_CPP_STANDARD -std=c++17"
    "PYBIND11_INSTALL ON CACHE BOOL"
)

CPMAddPackage("gh:NVIDIA/thrust#main")
find_package(Thrust REQUIRED CONFIG PATHS "${thrust_SOURCE_DIR}")
thrust_create_target(Thrust)
target_link_libraries(LPMP-BDD INTERFACE Thrust)

CPMAddPackage(
    NAME cereal
    GIT_TAG master
    GITHUB_REPOSITORY USCiLab/cereal
    OPTIONS
    "BUILD_DOC OFF"
    "BUILD_SANDBOX OFF"
    "SKIP_PERFORMANCE_COMPARISON" ON
)
target_include_directories(LPMP-BDD INTERFACE "${cereal_SOURCE_DIR}/include")

CPMAddPackage(
  NAME nlohmann_json
  VERSION 3.11.3
  GITHUB_REPOSITORY nlohmann/json
  OPTIONS
    "JSON_BuildTests OFF"
)
target_link_libraries(LPMP-BDD INTERFACE nlohmann_json::nlohmann_json)

#target_compile_options(LPMP-BDD INTERFACE -Wall -Wextra -Wpedantic -Werror)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("Compile with OpenMP")
    target_link_libraries(LPMP-BDD INTERFACE OpenMP::OpenMP_CXX)
else()
    message("No OpenMP found")
endif()

# OpenMP flags must be passed through nvcc to the host compiler
if(OpenMP_CXX_FOUND AND WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -Xlinker -fgomp")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)
