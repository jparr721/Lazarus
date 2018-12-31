set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(neural SHARED
  dqn.cc
  nn.cc
)

target_include_directories(neural
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(neural PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS neural
  EXPORT LazarusTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
)
