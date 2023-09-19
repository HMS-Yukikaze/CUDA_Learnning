# 📦Tracker 子项目的CMakeLists.txt
SET(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")

include_directories(bytetrack/include deepsort/include)

message(STATUS "${CMAKE_CURRENT_LIST_DIR}/bytetrack/src/BytekalmanFilter.cpp")

set(TRACKER_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/bytetrack/src/BytekalmanFilter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bytetrack/src/BYTETracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bytetrack/src/lapjv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bytetrack/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/FeatureTensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/hungarianoper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/kalmanfilter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/linear_assignment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/munkres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/nn_matching.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/track.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepsort/src/tracker.cpp
)

# 生成库
add_library(Tracker ${TRACKER_SOURCES})


target_include_directories(Tracker PUBLIC  
  include
)


target_link_libraries(Tracker PRIVATE 
  ${OpenCVRoot}/lib/opencv_world345d.lib
)





