cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(LazarusCorePreprocessing LANGUAGES CXX)

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for preprocessor libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for preprocessor executables")
set(INSTALL_INCLUDE_DIR inclue CACHE PATH "Installation direcotry for header files")

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/LazarusCorePreprocessing)
endif()

set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "installation directory for CMake files")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
      set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}")

add_subdirectory(kfold)
add_subdirectory(onehot)

export(TARGETS LazarusCorePreprocessing LazarusCorePreprocessingKFold LazarusCorePreprocessingOnehot
  FILE "${PROJECT_BINARY_DIR}/LazarusCorePreprocesingTargets.cmake")

export(PACKAGE LazarusCorePreprocessing)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
  "${INSTALL_INCLUDE_DIR}")

# for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(LazarusCorePreprocessingConfig.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusCorePreprocessingConfig.cmake" @ONLY)

# for the install tree
set(CONF_INCLUDE_DIRS "\${LAZARUSCORE_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(LazarusCorePreprocessingConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LazarusCorePreprocessingConfigVersion.cmake" @ONLY)

# for both
configure_file(LazarusCorePreprocessingConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/LazarusPreprocessingCoreConfigVersion.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LazarusCorePreprocessingConfig.cmake"
  "${PROJECT_BINARY_DIR}/LazarusCorePreprocessingConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT LazarusCorePreprocessingTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)
