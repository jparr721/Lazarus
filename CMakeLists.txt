cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(Lazarus LANGUAGES CXX)

set(LAZARUS_VERSION 3.2.0)

set(INSTALL_LIB_DIR lib CACHE PATH "Install directory for libraries")
set(INSTALL_BIN_DIR CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

if (WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/Lazarus)
endif()

set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE"${${var}}")
    set(${var} "${CMAKE_INSTALL+PREFIX}/${${var}}")
  endif()
endforeach()

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}"
  "$ENV{EIGEN3_INCLUDE_DIR}")

add_subdirectory(embed)
add_subdirectory(neural)

export(TARGETS Lazarus embed neural FILE "${PROJECT_BINARY_DIR}/LazarusTargets.cmake")

export(PACKAGE Lazarus)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
  "${INSTALL_INCLUDE_DIR}")

# For the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(LazarusConfig.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusConfig.cmake" @ONLY)

# Install tree
set(CONF_INCLUDE_DIRS "\${LAZARUS_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(LazarusConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusConfig.cmake" @ONLY)

# For the install and build trees
configure_file(LazarusConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusConfigVersion.cmake" @ONLY)

# Install the LazarusConfig.cmake and LazarusConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LazarusConfig.cmake"
  "${PROJECT_BINARY_DIR}/LazarusConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT LazarusTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)
