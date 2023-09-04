#include "Detect.h"

Detect::Detect()
{
}

Detect::~Detect()
{
}

void Detect::checkEnv()
{
	int driver_version = 0, runtime_version = 0;

	cudaDriverGetVersion(&driver_version);
	cudaRuntimeGetVersion(&runtime_version);

	std::cout << "Driver Version:" << driver_version << std::endl;
	std::cout << "Runtime Version: " << runtime_version << std::endl;
}

int Detect::setVideos(std::vector<std::string>& videoArrays)
{

	for each (std::string & var in videoArrays)
	{
		videos.push_back(var);
	}
	return 0;
}

int Detect::init(std::string& engine)
{

	return 0;
}

int Detect::capture_show(std::string& rtsp_address)
{

	// WrapData Wrap_img;
	cv::Mat frame;
	cv::VideoCapture cap;
	cap.open(rtsp_address);
	if (!cap.isOpened())
	{
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		auto start = std::chrono::system_clock::now();
		cap.read(frame);
		auto end = std::chrono::system_clock::now();
		std::cout << "cap.read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		// check if we succeeded
		if (frame.empty())
		{
			std::cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		//todo:Ô¤´¦Àí

	}
	cap.release();
	return 1;

}
