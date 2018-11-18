cmake_minimum_required(VERSION 3.9)
project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_library(lazarus SHARED IMPORTED)
set_property(TARGET lazarus PROPERTY IMPORTED LOCATION "../_install/lib/libLazarusCore.so.0.1.0")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ImageMagick REQUIRED COMPONENTS Magick++)

add_executable(tests
  test_perceptron.cc
)

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  ${ImageMagick_INCLUDE_DIRS})

add_executable(test
  "${PROJECT_SOURCE_DIR}/test_perceptron.cc"
  "${PROJECT_SOURCE_DIR}/main.cc"
)

target_link_libraries(test lazarus Eigen3::Eigen)
