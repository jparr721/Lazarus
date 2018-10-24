set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Boost REQUIRED)
find_package(ImageMagick REQUIRED COMPONENTS Magick++)

add_library(LazarusCore SHARED
  src/lazarus.cc
)

target_link_libraries(LazarusCore)

include_directories(${ImageMagick_INCLUDE_DIRS})

target_include_directories(LazarusCore
  PUBLIC
    $<INSTALL_INTERFACE:incude>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(LazarusCore PROPERTIES
  VERSION 0.1.0
  SOVERSION 0
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")

install(TARGETS LazarusCore
  EXPORT LazarusCoreTArgets
  RUNTIME DESTINATION "${INSTALL_BING_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
)
