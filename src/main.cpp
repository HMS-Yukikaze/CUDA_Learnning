#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda_utils.h"
#include "preprocess.h"
#include <opencv2/cudawarping.hpp>

void GetCudaVersion();


int main(void) {
	GetCudaVersion();

	cudaSetDevice(0);

	auto imgSize = BATCH_SIZE * INPUT_H * INPUT_W;
	auto dst = new float[2];

	cuda_preprocess_init(imgSize);
	std::vector<cv::Mat> arrays;
	//read img 
	cv::Mat pic = cv::imread("./resource/test.jpg");
	arrays.push_back(pic);

	cudaStream_t stream = nullptr;
	cudaStreamCreate(&stream);
	assert(stream != nullptr);

	//preprocess cols/width rows/height
	cuda_batch_preprocess(arrays, dst, INPUT_W, INPUT_H, stream);
	//cuda_preprocess(pic.data, pic.cols, pic.rows, dst, INPUT_W, INPUT_H, stream);

	//release
	cuda_preprocess_destroy();
	cudaStreamDestroy(stream);
	delete[] dst;
	return 0;
}

void GetCudaVersion() {
	int driver_version = 0, runtime_version = 0;

	cudaDriverGetVersion(&driver_version);
	cudaRuntimeGetVersion(&runtime_version);

	std::cout << "Driver Version:" << driver_version << std::endl;
	std::cout << "Runtime Version: " << runtime_version << std::endl;
}



