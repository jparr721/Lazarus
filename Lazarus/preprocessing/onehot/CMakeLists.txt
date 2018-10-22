set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LazarusLibCoreOnehot SHARED
  src/onehot.cc
)

target_link_libraries(LazarusLibCoreOnehot
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<UILD)NITERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusLibCoreOnehot PROPERTIES
  VERSION 0.1.0
  SOVERSION 1
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS LazarusLibCoreOnehot
  EXPORT LazarusCoreTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}/Lazarus/preprocessing/onehot" COMPONENT shlib
)
