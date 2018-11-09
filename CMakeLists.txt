cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(LazarusCore LANGUAGES CXX)

set(LAZARUSCORE_MAJOR_VERSION 0)
set(LAZARUSCORE_MINOR_VERSION 1)
set(LAZARUSCORE_PATCH_VERSION 0)
set (LAZARUSLIB_VERSION
  ${LAZARUSCORE_MAJOR_VERSION}.${LAZARUSCORE_MINOR_VERSION}.${LAZARUSCORE_PATCH_VERSION})

set(INSTALL_LIB_DIR lib CACHE PATH "Install directory for libraries")
set(INSTALL_BIN_DIR CACHE PATH "installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/LazarusCore)
endif()

set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "installation directory for CMake files")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
      set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

find_package(Boost 1.68.0 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}")

add_subdirectory(Lazarus/neural_network)
add_subdirectory(Lazarus/onehot)
add_subdirectory(Lazarus/kfold)
add_subdirectory(Lazarus/core)
add_subdirectory(Lazarus/svm)

# Add targets to build tree
export(TARGETS LazarusCore LazarusCoreNN LazarusCoreSVM
  FILE "${PROJECT_BINARY_DIR}/LazarusCoreTargets.cmake")

export(PACKAGE LazarusCore)

# Create LazarusCoreConfig and LazarusCoreConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
  "${INSTALL_INCLUDE_DIR}")

# build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(LazarusCoreConfig.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusCoreConfig.cmake" @ONLY)

# install tree
set(CONF_INCLUDE_DIRS "\${LAZARUS_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(LazarusCoreConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LazarusCoreConfig.cmake" @ONLY)

# Install and Build tree
configure_file(LazarusCoreConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusCoreConfigVersion.cmake" @ONLY)

# Install the LazarusCoreConfig.cmake and LazarusCoreConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LazarusCoreConfig.cmake"
  "${PROJECT_BINARY_DIR}/LazarusCoreConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use in the install tree
install(EXPORT LazarusCoreTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)
