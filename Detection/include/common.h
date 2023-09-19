#pragma once


#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include "NvInfer.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"
#include "yololayer.h"


class Logger : public nvinfer1::ILogger {
public:
	nvinfer1::ILogger::Severity reportableSeverity;

	explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
		reportableSeverity(severity)
	{
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
	{
		if (severity > reportableSeverity) {
			return;
		}
		switch (severity) {
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case nvinfer1::ILogger::Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "VERBOSE: ";
			break;
		}
		std::cerr << msg << std::endl;
	}
};

// get the size in byte of a TensorRT data type
__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType)
{
	switch ((int)dataType)
	{
		case int(nvinfer1::DataType::kFLOAT) :
			return 4;
		case int(nvinfer1::DataType::kHALF) :
			return 2;
		case int(nvinfer1::DataType::kINT8) :
			return 1;
		case int(nvinfer1::DataType::kINT32) :
			return 4;
		case int(nvinfer1::DataType::kBOOL) :
			return 1;
		default:
			return 4;
	}
}

// get the string of a TensorRT shape need  tensorrt 8.6.1 support
//__inline__ std::string shapeToString(nvinfer1::Dims32 dim)
//{
//	std::string output("(");
//	if (dim.nbDims == 0)
//	{
//		return output + std::string(")");
//	}
//	for (int i = 0; i < dim.nbDims - 1; ++i)
//	{
//		output += std::to_string(dim.d[i]) + std::string(", ");
//	}
//	output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
//	return output;
//}

// get the string of a TensorRT data type
__inline__ std::string dataTypeToString(nvinfer1::DataType dataType)
{
	switch (dataType)
	{
	case nvinfer1::DataType::kFLOAT:
		return std::string("FP32 ");
	case nvinfer1::DataType::kHALF:
		return std::string("FP16 ");
	case nvinfer1::DataType::kINT8:
		return std::string("INT8 ");
	case nvinfer1::DataType::kINT32:
		return std::string("INT32");
	case nvinfer1::DataType::kBOOL:
		return std::string("BOOL ");
	default:
		return std::string("Unknown");
	}
}

// get the string of a TensorRT data format
__inline__ std::string formatToString(nvinfer1::TensorFormat format)
{
	switch (format)
	{
	case nvinfer1::TensorFormat::kLINEAR:
		return std::string("LINE ");
	case nvinfer1::TensorFormat::kCHW2:
		return std::string("CHW2 ");
	case nvinfer1::TensorFormat::kHWC8:
		return std::string("HWC8 ");
	case nvinfer1::TensorFormat::kCHW4:
		return std::string("CHW4 ");
	case nvinfer1::TensorFormat::kCHW16:
		return std::string("CHW16");
	case nvinfer1::TensorFormat::kCHW32:
		return std::string("CHW32");
	case nvinfer1::TensorFormat::kHWC:
		return std::string("HWC  ");
	//case nvinfer1::TensorFormat::kDLA_LINEAR:
	//	return std::string("DLINE");
	//case nvinfer1::TensorFormat::kDLA_HWC4:
	//	return std::string("DHWC4");
	//case nvinfer1::TensorFormat::kHWC16:
	//	return std::string("HWC16");
	default:
		return std::string("None ");
	}
}

// get the string of a TensorRT layer kind
__inline__ std::string layerTypeToString(nvinfer1::LayerType layerType) {}



inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
	int size = 1;
	for (int i = 0; i < dims.nbDims; i++) {
		size *= dims.d[i];
	}
	return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
	switch (dataType) {
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kINT8:
		return 1;
	case nvinfer1::DataType::kBOOL:
		return 1;
	default:
		return 4;
	}
}

namespace det {
	static constexpr int BATCH_SIZE = 1;
	static constexpr int CHECK_COUNT = 3;

	static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
	static constexpr int INPUT_W = 640;
	
	static constexpr int LOCATIONS = 4;
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;

	static constexpr int MAX_IMAGE_INPUT_SIZE_THRESH = 3000;

	static char INPUT_BLOB_NAME[] = "data";
	static char OUTPUT_BLOB_NAME[] = "prob";

	static constexpr float CONF_THRESH = 0.5;
	static constexpr float NMS_THRESH  = 0.4;

	struct Binding {
		size_t         size = 1;
		size_t         dsize = 1;
		nvinfer1::Dims dims;
		std::string    name;
	};

	//struct Object {
	//	cv::Rect_<float> rect;
	//	int              label = 0;
	//	float            prob = 0.0;
	//};

	struct alignas(float) Detection {
		//center_x center_y w h
		float bbox[LOCATIONS];
		float conf;  // bbox_conf * cls_conf
		float class_id;
	};

	struct YoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT * 2];
	};

	struct PreParam {
		float ratio = 1.0f;
		float dw = 0.0f;
		float dh = 0.0f;
		float height = 0;
		float width = 0;
	};
}  // namespace det