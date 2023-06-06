#define __CL_ENABLE_EXCEPTIONS

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include <CL/cl.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

bool less_by_y(const cv::Point &lhs, const cv::Point &rhs) {
  return lhs.y > rhs.y;
}

struct Point {
  double x, y, z;
};

static std::string readStringFromFile(const std::string &filename) {
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    printf("Couldn't open file '%s'!\n", filename.c_str());
    return "";
  }

  size_t filesize = 0;
  is.seekg(0, std::ios::end);
  filesize = (size_t)is.tellg();
  is.seekg(0, std::ios::beg);

  std::string source{std::istreambuf_iterator<char>(is),
                     std::istreambuf_iterator<char>()};

  return source;
}

int main(void) {
  int width = 750, height = 960;
  float *output_buffer = (float *)malloc(sizeof(float) * 3 * height);
  cl_int err = CL_SUCCESS;

  cv::Mat img_cv = cv::imread("imgs/input.jpg", cv::IMREAD_GRAYSCALE);
   auto end = std::chrono::system_clock::now();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<Point> points;

  // for (int i = 0; i < img_cv.rows; ++i) {
  //   for (int j = 0; j < img_cv.cols; ++j) {
  //     uchar pixelValue = img_cv.at<uchar>(i, j);
  //     if (pixelValue) {
  //       std::cout << j << " " << i << std::endl;
  //       break;
  //     }
  //   }
  // }
  double angle = 1.125;
  double x_size = 514;

  auto start = std::chrono::system_clock::now();
  cv::findContours(img_cv, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE,
                   cv::Point(0, 0));

  for (size_t j = 0; j < contours.size(); ++j) {
    std::vector<cv::Point>::iterator result =
        std::min_element(contours[j].begin(), contours[j].end(), less_by_y);
    contours[j].erase(result, contours[j].end());
  }

  for (size_t k = 0; k < contours.size(); ++k) {
    for (size_t m = 0; m < contours[k].size(); ++m) {
      Point point;
      point.x = (contours[k][m].x - x_size) * 2. * cos(angle * 3.1415 / 180);
      point.y = (contours[k][m].x - x_size) * 2. * sin(angle * 3.1415 / 180);
      point.z = contours[k][m].y;

      points.push_back(point);
    }
  }
 
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  printf("CPU Execution time is: %0.3f milliseconds \n", elapsed / 1000000.0);
  try {

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cout << "Platform size 0\n";
      return -1;
    }

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    cl::Context context(CL_DEVICE_TYPE_ALL, properties);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    std::string kernel_path = "kernels/contour.cl";
    std::string kernel_str = readStringFromFile(kernel_path);
    cl::Program::Sources source(
        1, std::make_pair(kernel_str.c_str(), kernel_str.size()));
    cl::Program program_ = cl::Program(context, source);
    program_.build(devices);

    cl::Kernel kernel(program_, "contour", &err);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE,
                           &err);

    cl::Buffer buffer_src(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          width * height);
    cl::Buffer buffer_dst(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          3 * height * sizeof(float));

    queue.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, width * height,
                             img_cv.data);
    float angle = 1.125f;
    kernel.setArg(0, buffer_src);
    kernel.setArg(1, buffer_dst);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, angle);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(height),
                               cl::NDRange(64), NULL, &event);

    event.wait();

    queue.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, 3 * height * sizeof(float),
                            output_buffer);
    queue.finish();

    double nanoSeconds =
        double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
               event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    printf("GPU Execution time is: %0.3f milliseconds \n",
           nanoSeconds / 1000000.0);

    std::cout << std::setprecision(3);
    int cnt = 0;
    for (int p = 0; p < height; ++p) {
      std::cout << "v: " << output_buffer[3 * p] << " "
                << output_buffer[3 * p + 1] << " " << output_buffer[3 * p + 2]
                << std::endl;
      if (output_buffer[3 * p + 2] > 0) {
        cnt++;
      }
    }
    std::cout << "Points:" << cnt << std::endl;
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  // stbi_write_jpg("imgs/lane_out.jpg", width, height, 1, output_img, 100);

  // cpuTest(img, output_img, width, height);
  return EXIT_SUCCESS;
}