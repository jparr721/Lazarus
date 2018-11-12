set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Boost REQUIRED)
find_package(ImageMagick REQUIRED COMPONENTS Magick++)

add_executable(tests
  test_perceptron.cc
)

include_directories(${ImageMagick_INCLUDE_DIRS})

target_link_libraries(tests
  LazarusCoreNN
  LazarusCoreUtil
)

install(TARGETS tests
  EXPORT LazarusTestTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
