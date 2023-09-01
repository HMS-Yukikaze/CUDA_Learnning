#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

static constexpr int BATCH_SIZE=1;
static constexpr int INPUT_H = 1920;  // yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 1080;


#endif  // TRTX_CUDA_UTILS_H_