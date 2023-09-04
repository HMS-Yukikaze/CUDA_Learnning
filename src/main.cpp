#include <cuda_runtime_api.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "cuda_utils.h"
#include "preprocess.h"
#include "opencv2/opencv.hpp"


static constexpr int kMaxInputImageSize = 1920 * 1080;
std::vector<cv::Mat> imgLists;

int main(void)
{
	//◊º±∏≤‚ ‘ ”∆µ
	std::string rtsp1 = "E:\\WorkSpace\\C++\\rtsp-cuda\\x64\\Release\\Cloth\\Cloth.mp4";
	std::string rtsp2 = "E:\\WorkSpace\\C++\\rtsp-cuda\\x64\\Release\\Cloth\\Clothbei.mp4";


	std::vector<std::string> rtspList = { rtsp1,rtsp2 };//store the whole rtsp address
	// Deserialize the engine from file


	return 0;
}






