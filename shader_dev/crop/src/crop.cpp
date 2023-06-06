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

void cpuTest(const unsigned char *src, unsigned char *dst, int width,
             int height, int crop_width, int crop_height, int x_offset,
             int y_offset) {
  auto start = std::chrono::system_clock::now();
  for (int src_row = 0; src_row < height; src_row++) {
    for (int src_col = 0; src_col < width; src_col++) {
      if (src_row >= y_offset && src_row < (y_offset + crop_height) &&
          src_col >= x_offset && src_col < (x_offset + crop_width)) {
        int dst_row = src_row - y_offset;
        int dst_col = src_col - x_offset;

        int src_index = (src_row * width + src_col) * 3;
        int dst_index = (dst_row * crop_width + dst_col) * 3;

        dst[dst_index] = src[src_index];
        dst[dst_index + 1] = src[src_index + 1];
        dst[dst_index + 2] = src[src_index + 2];
      }
    }
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  printf("CPU Execution time is: %0.3f milliseconds \n", elapsed / 1000000.0);
}

int main(void) {

  int width, height, channels;
  unsigned char *img =
      stbi_load("imgs/input.jpg", &width, &height, &channels, 0);

  int crop_width = 885 - 135, crop_height = height;
  int crop_xoffset = 135, crop_yoffset = 0;

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

    std::string kernel_path = "kernels/crop.cl";
    std::string kernel_str = readStringFromFile(kernel_path);
    cl::Program::Sources source(
        1, std::make_pair(kernel_str.c_str(), kernel_str.size()));
    cl::Program program_ = cl::Program(context, source);
    program_.build(devices);

    cl::Kernel kernel(program_, "crop", &err);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE,
                           &err);

    cl::Buffer buffer_src(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          channels * width * height);
    cl::Buffer buffer_dst(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          channels * crop_width * crop_height);

    unsigned char *output_img = (unsigned char *)queue.enqueueMapBuffer(
        buffer_dst, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
        channels * crop_width * crop_height);

    queue.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, channels * width * height,
                             img);

    kernel.setArg(0, buffer_src);
    kernel.setArg(1, buffer_dst);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, crop_width);
    kernel.setArg(5, crop_height);
    kernel.setArg(6, crop_xoffset);
    kernel.setArg(7, crop_yoffset);

    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange,
        cl::NDRange(width * height), cl::NDRange(64),
        NULL, &event);

    event.wait();

    queue.enqueueReadBuffer(buffer_dst, CL_TRUE, 0,
                            channels * crop_width * crop_height, output_img);
    queue.finish();

    double nanoSeconds =
        double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
               event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    printf("GPU Execution time is: %0.3f milliseconds \n",
           nanoSeconds / 1000000.0);

    stbi_write_jpg("imgs/gpu_out.jpg", crop_width, crop_height, channels,
                   output_img, 100);

    cpuTest(img, output_img, width, height, crop_width, crop_height,
            crop_xoffset, crop_yoffset);
    stbi_write_jpg("imgs/cpu_out.jpg", crop_width, crop_height, channels,
                   output_img, 100);
    return EXIT_SUCCESS;
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }
}