cmake_minimum_required(VERSION 3.1)

SET(CMAKE_TOOLCHAIN_FILE "D:/1_workSpace/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(OpenCV_CUDA LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()  

# detect os
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring on/for Windows")
endif()

# Set CUDA architectures (change as needed)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


message("Opencv_Cuda: $ENV{Opencv_Cuda}")


# set any CMAKE variables here.  
  
# Find packages go here.  

file(GLOB_RECURSE ALL_SOURCES
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/*.c"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"    
)

# Filter out files from the build directory
#list(FILTER ALL_SOURCES EXCLUDE REGEX ".*build/.*")

source_group(TREE "${CMAKE_SOURCE_DIR}" FILES ${ALL_SOURCES})


add_executable(OpenCV_CUDA
              ${ALL_SOURCES}
              )

#指定项目头文件路径
target_include_directories(OpenCV_CUDA PRIVATE
    $ENV{Opencv_Cuda}/include
    ${CMAKE_SOURCE_DIR}/include
)

#指定项目依赖库路径
target_link_libraries(OpenCV_CUDA PRIVATE
    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_core470d.lib
    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_imgproc470d.lib
    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_imgcodecs470d.lib
    cudart.lib
)


enable_testing()
add_subdirectory(test)