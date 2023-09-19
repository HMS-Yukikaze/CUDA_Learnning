cmake_minimum_required(VERSION 3.1)

SET(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(YOLOVX LANGUAGES CXX CUDA)

# add to cmake module path
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()  

# detect os
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring on/for Windows")
endif()

# Set CUDA architectures (change as needed) https://developer.nvidia.com/cuda-gpus
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

#message("Opencv_Cuda: $ENV{Opencv_Cuda}")

# set any CMAKE variables here.  
SET(OpenCVRoot E:/WorkSpace/C++/3rd/opencv/opencv3.4.5)
SET(TsrtRoot E:/WorkSpace/C++/3rd/TensorRT-7.2.1.6)  

# Find packages go here.  

#https://github.com/wang-xinyu/tensorrtx/issues/666
add_definitions(-DAPI_EXPORTS) 

# 添加子目录 Detection 和 Tracker
add_subdirectory(Detection)
add_subdirectory(Tracker)

# 添加可执行文件
add_executable(YOLOVX
    src/main.cpp
    src/manager.cpp
)


#启用 C++17
target_compile_features(YOLOVX PUBLIC cxx_std_17)
set_target_properties(YOLOVX PROPERTIES CXX_EXTENSIONS OFF)


#指定项目头文件路径
#$ENV{BoostRootDir}
target_include_directories(YOLOVX PRIVATE 
    $ENV{BoostRootDir}   
    ${OpenCVRoot}/include
    ${CMAKE_SOURCE_DIR}/include
    ${TsrtRoot}/include    
)

#指定项目依赖库路径
#$ENV{BoostRootDir}/stage/lib/*.lib
target_link_libraries(YOLOVX PRIVATE    
    ${OpenCVRoot}/lib/opencv_world345d.lib
    ${TsrtRoot}/lib/nvinfer.lib
    ${TsrtRoot}/lib/nvinfer_plugin.lib
    cudart.lib   
    Detection
    Tracker
)



add_subdirectory(test) 

