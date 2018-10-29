set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LazarusCoreKFold SHARED
  src/kfold.cc
)

#target_link_libraries(LazarusCorePreprocessingKFold LazarusCorePreprocessing)

target_include_directories(LazarusCoreKFold
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusCoreKFold PROPERTIES
  VERSION 0.1.0
  SOVERSION 0
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
install(TARGETS LazarusCoreKFold
  EXPORT LazarusCoreTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}/Lazarus/kfold" COMPONENT shlib
)
