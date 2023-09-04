#include "gtest/gtest.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"

TEST(A, B) {
  test();

  SUCCEED();
}

int
main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
/*

 ┣ 📂src
 ┃ ┣ 📜main.cpp
 ┣ 📂test
 ┃ ┣ 📜CMakeLists.txt
 ┃ ┣ 📜kernels.cu
 ┃ ┣ 📜kernels.cuh
 ┃ ┗ 📜test.cpp
*/