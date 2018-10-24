set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LazarusCoreSVM SHARED
  src/svm.cc
)

target_link_libraries(LazarusCoreSVM)

target_include_directories(LazarusCoreSVM
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusCoreSVM PROPERTIES
  VERSION 0.1.0
  SOVERSION 1
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS LazarusCoreSVM
  EXPORT LazarusCoreTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}/Lazarus/core" COMPONENT shlib
)
