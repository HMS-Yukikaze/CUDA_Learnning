#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include "preprocess.h"
#include "common.h"




class Detect
{
public:
	Detect();

	~Detect();

	void checkEnv();

	int init(std::string& engine);
	
	int addTask(std::string& videoSource);

	int TasksRun();

	//算法分析
	int Analysis();

	//预处理
	int preProcess(cv::Mat& img);

	//推理
	void doinference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize);

	//非极大值抑制算法
	void nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh);


private:
	void release();

	float iou(float lbox[4], float rbox[4]);
	
	bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);
	
	int capture_show(std::string& rtsp_address);
private:

	std::vector<std::string> videos;
	std::vector<std::thread> threads;
	std::list<cv::Mat> imglist;

	int inputIndex = 0;//输入张量的绑定索引
	int outputIndex = 0;//输出张量的绑定索引

	float* buffers[2];                     //存储输入和输出的设备缓冲区地址
	uint8_t* img_host;				       //指向输入数据的指针
	uint8_t* img_device;				   //指向输出数据的指针	
	nvinfer1::ICudaEngine* cuda_engine = nullptr;
	nvinfer1::IRuntime* cuda_runtime = nullptr;
	nvinfer1::IExecutionContext* cuda_context = nullptr; // TensorRT执行上下文
	cudaStream_t cuda_stream = nullptr;
	Logger gLogger;
};

