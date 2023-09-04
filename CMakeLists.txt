cmake_minimum_required(VERSION 3.1)

SET(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
SET(OpenCVRoot E:/WorkSpace/C++/3rd/opencv/opencv3.4.5)
SET(TsrtRoot E:/WorkSpace/C++/3rd/TensorRT-8.6.1.6)


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
    $ENV{BoostRootDir}
    ${OpenCVRoot}/include
    ${CMAKE_SOURCE_DIR}/include
    ${TsrtRoot}/include    
)

#指定项目依赖库路径
target_link_libraries(OpenCV_CUDA PRIVATE
    $ENV{BoostRootDir}/stage/lib/*.lib
    ${OpenCVRoot}/lib/opencv_world345d.lib
    ${TsrtRoot}/lib/nvinfer.lib
    ${TsrtRoot}/lib/nvinfer_plugin.lib
    cudart.lib
)

add_subdirectory(test)