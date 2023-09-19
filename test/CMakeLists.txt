enable_testing()
SET(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Find Google Test package
find_package(GTest REQUIRED)

message(STATUS: "test moudle gen start...")

include_directories(
    ${GTEST_INCLUDE_DIRS}
    )

set(TEST_SOURCES    
    kernels.cu
    test.cpp
)

# Add the test executable
add_executable(mTest ${TEST_SOURCES})

#Link Google Test and CUDA libraries
target_link_libraries(mTest PRIVATE GTest::GTest cudart)