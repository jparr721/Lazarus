set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LazarusCoreNN SHARED
  src/multilayer_perceptron.cc
)

target_link_libraries(LazarusCoreNN)

target_include_directories(LazarusCoreNN
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusCoreNN PROPERTIES
  VERSION 0.1.0
  SOVERSION 1
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS LazarusCoreNN
  EXPORT LazarusCoreTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}/Lazarus/core" COMPONENT shlib
)
