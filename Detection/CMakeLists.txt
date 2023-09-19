# 📦Detection 子项目的CMakeLists.txt
SET(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(DETECTION_SOURCES
    src/Detect.cpp
    src/preprocess.cu
    src/yololayer.cu
)

# 生成库
add_library(Detection ${DETECTION_SOURCES})


target_include_directories(Detection PUBLIC 
  ${OpenCVRoot}/include
  ${TsrtRoot}/include    
  include
)


target_link_libraries(Detection PRIVATE 
  ${OpenCVRoot}/lib/opencv_world345d.lib
  ${TsrtRoot}/lib/nvinfer.lib
  ${TsrtRoot}/lib/nvinfer_plugin.lib
  cudart.lib
)





