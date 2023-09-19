#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "bounded_buffer.hpp"
#include "Detect.h"


class Manager
{
private:
	/* data */
	std::string engine;
	std::shared_ptr<Detect> detector;
	bounded_buffer<cv::Mat> buf;
public:
	Manager(std::string _file, size_t _sz = 1000);
	~Manager();
};


