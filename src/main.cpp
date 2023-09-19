#include <cuda_runtime_api.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "cuda_utils.h"
#include "preprocess.h"
#include "Detect.h"


static constexpr int kMaxInputImageSize = 1920 * 1080;
std::vector<cv::Mat> imgLists;

int main(void)
{
	//准备测试视频
	std::string rtsp1 = "E:\\WorkSpace\\C++\\rtsp-cuda\\x64\\Release\\Cloth\\Cloth.mp4";
	std::string rtsp2 = "E:\\WorkSpace\\C++\\rtsp-cuda\\x64\\Release\\Cloth\\Clothbei.mp4";

	//准备测试engine
	std::string fEngine("C:\\Users\\admin\\Desktop\\更新\\Engine\\100100002\\fire_smokeV10.5.engine");

	//测试文件是否存在


	Detect infer;

	infer.addTask(rtsp1);
	infer.addTask(rtsp2);
	// Deserialize the engine from file
	infer.init(fEngine);
	//infer.init(fEngine);
	//infer.init(fEngine);
	infer.TasksRun();

	return 0;
}






