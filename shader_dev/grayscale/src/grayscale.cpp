#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>


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

static void cpuTest(unsigned char *src, unsigned char *dst, int width,
                    int height) {
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < width * height; ++i) {
    float r = src[3 * i];
    float g = src[3 * i + 1];
    float b = src[3 * i + 2];
    dst[i] =  (r * 0.299f + g * 0.587f + b * 0.114) >= 200.0f ? 255 : 0;
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  printf("CPU Execution time is: %0.3f milliseconds \n",
         elapsed / 1000000.0);
}

int main(void) {



  int crop_width, crop_height;
  int crop_xoffset, crop_yoofset;


  int width, height, channels;
  unsigned char *img =
      stbi_load("imgs/lane.jpg", &width, &height, &channels, 0);
  unsigned char *output_img = (unsigned char *)malloc(width * height);
  cl_int err = CL_SUCCESS;
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

    std::string kernel_path = "kernels/grayscale.cl";
    std::string kernel_str = readStringFromFile(kernel_path);
    cl::Program::Sources source(
        1, std::make_pair(kernel_str.c_str(), kernel_str.size()));
    cl::Program program_ = cl::Program(context, source);
    program_.build(devices);

    cl::Kernel kernel(program_, "grayscale", &err);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE,
                           &err);

    cl::Buffer buffer_src(context, CL_MEM_READ_WRITE,
                          channels * width * height);
    cl::Buffer buffer_dst(context, CL_MEM_READ_WRITE, width * height);

    queue.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, channels * width * height,
                             img);

    kernel.setArg(0, buffer_src);
    kernel.setArg(1, buffer_dst);
    kernel.setArg(2, width);
    kernel.setArg(3, height);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width * height / 10), cl::NDRange(4),
                               NULL, &event);

    event.wait();

    queue.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, width * height, output_img);
    queue.finish();

    double nanoSeconds =
        double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
               event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    printf("GPU Execution time is: %0.3f milliseconds \n",
           nanoSeconds / 1000000.0);
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
  stbi_write_jpg("imgs/lane_out.jpg", width, height, 1, output_img, 100);

  cpuTest(img, output_img, width, height);
  return EXIT_SUCCESS;
}