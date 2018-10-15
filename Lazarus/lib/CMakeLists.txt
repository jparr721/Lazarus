cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(LazarusCoreLib VERSION 0.1.0 LANGUAGES CXX)

set(LAZARUSLIB_MAJOR_VERSION 0)
set(LAZARUSLIB_MINOR_VERSION 1)
set(LAZARUSLIB_PATCH_VERSION 0)
set (LAZARUSLIB_VERSION
  ${LAZARUSLIB_MAJOR_VERSION}.${LAZARUSLIB_MINOR_VERSION}.${LAZARUSLIB_PATCH_VERSION})

set(INSTALL_LIB_DIR lib CACHE PATH "Install directory for libraries")

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}")

add_subdirectory(lib/kfold)

export (TARGETS LazarusANN LazarusCoreLib LazarusKfold
  FILE "${PROJECT_BINARY_DIR}/LazarusCoreTargets.cmake")

export (PACKAGE LazarusCoreLib)

