cmake_minimum_required(VERSION 3.5)
project(scan_gpu)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(Vuh REQUIRED)

find_package( Boost COMPONENTS program_options REQUIRED )

find_package( OpenCV REQUIRED )

find_package(JPEG REQUIRED)

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SCAN_GPU_SOURCE_FILES
    src/scan_gpu.cpp
)

add_executable(scan_gpu ${SCAN_GPU_SOURCE_FILES})
target_link_libraries(scan_gpu PUBLIC vuh::vuh ${Boost_LIBRARIES} ${OpenCV_LIBS}  ${JPEG_LIBRARY})
target_include_directories(scan_gpu PRIVATE ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR} "include" "kernels")


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(scan_gpu PRIVATE -fPIC -Wall -Wextra -Wno-unused-para -O3)
endif()