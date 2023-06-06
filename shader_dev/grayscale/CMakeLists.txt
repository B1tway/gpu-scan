cmake_minimum_required(VERSION 3.5)
project(grayscale)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(OpenCL)
FIND_PATH(HPP_FOUND CL/cl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  find_package(OpenCL REQUIRED)

  set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} "${SDK_ROOT}/include")
  FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()

if(NOT OpenCL_FOUND)
  message(FATAL_ERROR "Could not find OpenCL include/libs. Set OCL_ROOT to your OpenCL SDK")
else()
  message(STATUS "Selected OpenCL includes from ${OpenCL_INCLUDE_DIRS}")
  message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")
endif()

# extra libs for static library
if(UNIX AND ${OpenCL_LIBRARIES} MATCHES "libOpenCL.a")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} pthread dl)
endif()

FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )


set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GRAYSCALE_SOURCE_FILES
    src/grayscale.cpp
)

add_executable(grayscale ${GRAYSCALE_SOURCE_FILES})
target_link_libraries(grayscale ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(grayscale PRIVATE ${OpenCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} "include" "kernels")
target_compile_definitions(grayscale PRIVATE CL_TARGET_OPENCL_VERSION=120)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(grayscale PRIVATE -fPIC -Wall -Wextra -Wno-unused-para)
endif()