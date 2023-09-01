#define BOOST_TEST_MODULE MyTest
#include <boost/test/included/unit_test.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/cudawarping.hpp>
#include <chrono>
//#include "vld.h"

void prepareInput_gpu(cv::Mat& img, int in_w, int in_h);
static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
#if 1
BOOST_AUTO_TEST_CASE(test_prepareInput)
{


    // Create a sample input image (you can replace this with your test image)
    //cv::Mat inputImage(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    // Define expected output dimensions
    const int INPUT_W = 416;
    const int INPUT_H = 400;

    cv::Mat pic = cv::imread("./resource/test.jpg");

    // Call the prepareInput function
    auto out = preprocess_img(pic, INPUT_W, INPUT_H);

    pic.release();
    BOOST_CHECK_EQUAL(out.cols, INPUT_W);
   BOOST_CHECK_EQUAL(out.rows, INPUT_H);
}
#endif

void prepareInput_gpu(cv::Mat& img, int in_w, int in_h)
{
    int w, h, x, y;
    float r_w = in_w / (img.cols * 1.0);
    float r_h = in_h / (img.rows * 1.0);

    if (r_h > r_w) {
        w = in_w;
        h = r_w * img.rows;
        x = 0;
        y = (in_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = in_h;
        x = (in_w - w) / 2;
        y = 0;
    }

    // Create GPU Mats for resizing and output
    cv::cuda::GpuMat d_img(img);
    cv::cuda::GpuMat d_re;

    // Resize on GPU
    cv::cuda::resize(d_img, d_re, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    // Download the result from GPU to CPU
    cv::Mat re;
    d_re.download(re);

    cv::Mat out(in_h, in_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

}

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

#if 0
int main() {
    
    const int INPUT_W = 416;
    const int INPUT_H = 400;

    cv::Mat pic = cv::imread("./resource/test.jpg");

    auto start = std::chrono::system_clock::now();
    // Call the prepareInput function
    prepareInput_gpu(pic, INPUT_W, INPUT_H);
    auto end = std::chrono::system_clock::now();
    
    auto dura = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "time cost:"<<dura << std::endl;
    pic.release();
    return 0;
}
#endif
