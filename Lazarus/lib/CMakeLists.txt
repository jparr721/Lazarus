cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(lib VERSION 0.1.0 LANGUAGES CXX)

add_library(LazarusCoreLib SHARED
  src/kfold.cc
)

target_link_libraries(LazarusCore)

target_include_directories(LazarusCoreLib
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOUCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusCoreLib PROPERTIES
  VERSION 0.1.0
  SOVERSION 1
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS LazarusCoreLib
  EXPORT LazarusCoreTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}/Lazarus/core" COMPONENT shlib
)
