
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(mtest test.cpp)

#指定项目头文件路径
target_include_directories(mtest PRIVATE
    $ENV{Opencv_Cuda}/include
)

#指定项目依赖库路径
target_link_libraries(mtest 
                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_core470d.lib
                    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_imgproc470d.lib
                    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_imgcodecs470d.lib
                    $ENV{Opencv_Cuda}/x64/vc16/lib/opencv_cudawarping470d.lib
                    )