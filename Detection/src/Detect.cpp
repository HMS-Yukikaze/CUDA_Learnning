#include "Detect.h"
#include <cuda_utils.h>


static const int OUTPUT_SIZE = det::MAX_OUTPUT_BBOX_COUNT * sizeof(det::Detection) / sizeof(float) + 1;

using namespace nvinfer1;

Detect::Detect()
{
	buffers[0] = nullptr;
	buffers[1] = nullptr;
}

Detect::~Detect()
{
	release();
}

void Detect::checkEnv()
{
	int driver_version = 0, runtime_version = 0;

	cudaDriverGetVersion(&driver_version);
	cudaRuntimeGetVersion(&runtime_version);

	std::cout << "Driver Version:" << driver_version << std::endl;
	std::cout << "Runtime Version: " << runtime_version << std::endl;
}

int Detect::addTask(std::string& videoArray)
{
	videos.push_back(videoArray);
	return 0;
}

int Detect::TasksRun()
{
	std::vector<std::thread> threads;

	for each (auto url in videos)
	{
		threads.push_back(std::thread(&Detect::Analysis,this));
	}
	
	for(std::thread& th : threads) {
		th.join();
	}

	return 0;
}

int Detect::Analysis()
{
	auto thd_ID = std::this_thread::get_id();

	std::printf("预处理:%d\n", thd_ID);

	std::printf("推理:%d\n", thd_ID);

	std::printf("画框%d\n", thd_ID);
	return 0;
}

int Detect::init(std::string& engine)
{
	cudaSetDevice(0);

	// deserialize the .engine and run inference
	std::ifstream file(engine, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << engine << " error!" << std::endl;
		return -1;
	}
	char* trtModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();

	static float prob[det::BATCH_SIZE * OUTPUT_SIZE];
	try
	{
		cuda_runtime = createInferRuntime(gLogger);
		assert(cuda_runtime != nullptr);
		cuda_engine = cuda_runtime->deserializeCudaEngine(trtModelStream, size);
		assert(cuda_engine != nullptr);
		cuda_context = cuda_engine->createExecutionContext();
		assert(cuda_context != nullptr);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
	std::cout << "finish load engine" << std::endl;

	delete[] trtModelStream;

	assert(cuda_engine->getNbBindings() == 2);
	
	inputIndex = cuda_engine->getBindingIndex(det::INPUT_BLOB_NAME);
	outputIndex = cuda_engine->getBindingIndex(det::OUTPUT_BLOB_NAME);
	assert(inputIndex == 0);
	assert(outputIndex == 1);
	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc((void**)&buffers[0], det::BATCH_SIZE * 3 * det::INPUT_H * det::INPUT_W * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&buffers[1], det::BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
	
	// Create stream
	CUDA_CHECK(cudaStreamCreate(&cuda_stream));

	// prepare input data cache in pinned memory 
	CUDA_CHECK(cudaMallocHost((void**)&img_host, det::MAX_IMAGE_INPUT_SIZE_THRESH * 3));
	// prepare input data cache in device memory
	CUDA_CHECK(cudaMalloc((void**)&img_device, det::MAX_IMAGE_INPUT_SIZE_THRESH * 3));




	return 0;
}

int Detect::preProcess(cv::Mat& img)
{
	size_t  inputSize = img.cols * img.rows * 3;
	//copy data to pinned memory
	memcpy(img_host, img.data, inputSize);
	//copy data to device memory
	CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, inputSize, cudaMemcpyHostToDevice, cuda_stream));
	preprocess_kernel_img(img_device, img.cols, img.rows, (float*)buffers[inputIndex], det::INPUT_W, det::INPUT_H, cuda_stream);
	return 0;
}

void Detect::doinference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize)
{
	CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * det::INPUT_H * det::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));//add
	context.enqueue(batchSize, buffers, stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}

void Detect::nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh)
{
	int det_size = sizeof(Yolo::Detection) / sizeof(float);
	std::map<float, std::vector<Yolo::Detection>> m;
	for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Yolo::Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), 
			[](const Yolo::Detection& a, const Yolo::Detection& b) {
				return a.conf > b.conf;//降序
			});
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}

}

void Detect::release()
{
	CUDA_CHECK(cudaFree(buffers[inputIndex]));
	CUDA_CHECK(cudaFree(buffers[outputIndex]));

	if (cuda_context) {
		cuda_context->destroy();
	}
	if (cuda_engine) {
		cuda_engine->destroy();
	}
	if (cuda_runtime) {
		cuda_runtime->destroy();
	}

}

float Detect::iou(float lbox[4], float rbox[4])
{
	float interBox[] = {
	  (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
	  (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
	  (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
	  (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool Detect::cmp(const Yolo::Detection& a, const Yolo::Detection& b)
{
	return a.conf > b.conf;
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
		//todo:预处理

	}
	cap.release();
	return 1;

}
