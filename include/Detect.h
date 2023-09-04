#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>


class Detect
{
public:
	Detect();
	~Detect();

	void checkEnv();

	int setVideos(std::vector<std::string>& videoArrays);

	int init(std::string& engine);

	int TasksRun();

	//Ԥ�����߳�
	int preProcess();

	//�����߳�

	//�����߳�

	//��ʾ�߳�
	int show();
private:
	int capture_show(std::string& rtsp_address);
private:
	int srcSize;
	
	std::vector<std::string> videos;
	std::vector<std::thread> threads;
	std::list<cv::Mat> imglist;
};

