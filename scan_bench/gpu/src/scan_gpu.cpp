#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <arm_neon.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <jpeglib.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

bool less_by_y(const cv::Point &lhs, const cv::Point &rhs) {
  return lhs.y > rhs.y;
}

struct Point {
  double x, y, z;
};

const int LEFT = 135;
const int RIGHT = 885;
const double x_size = 514; // 526

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

void readJpeg(const char *filename, unsigned char *buffer, int width,
              int height) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    // Обработка ошибки открытия файла
    return;
  }

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);

  // Установка параметров декомпрессии
  cinfo.out_color_space = JCS_RGB;
  cinfo.output_width = width;
  cinfo.output_height = height;
  cinfo.scale_num = 1;
  cinfo.scale_denom = 1;
  cinfo.dct_method = JDCT_FASTEST;

  jpeg_start_decompress(&cinfo);

  // Чтение строк изображения
  JSAMPARRAY row_buffer = (*cinfo.mem->alloc_sarray)(
      (j_common_ptr)&cinfo, JPOOL_IMAGE,
      cinfo.output_width * cinfo.output_components, 1);
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, row_buffer, 1);
    memcpy(buffer + (cinfo.output_scanline - 1) * width * 3, row_buffer[0],
           width * 3);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(file);
}

int main(void) {

  std::ofstream result_stream;
  result_stream.open("/home/rock/git/scan_bench/gpu/build/result_gpu.obj");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cout << "Platform size 0\n";
    return -1;
  }
  cl_int err = CL_SUCCESS;
  cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
  cl::Context context(CL_DEVICE_TYPE_ALL, properties);

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  std::string crop_kernel_path = "kernels/crop.cl";
  std::string crop_kernel_str = readStringFromFile(crop_kernel_path);
  cl::Program::Sources crop_source(
      1, std::make_pair(crop_kernel_str.c_str(), crop_kernel_str.size()));
  cl::Program crop_program = cl::Program(context, crop_source);
  crop_program.build(devices);

  cl::Kernel crop_kernel(crop_program, "crop", &err);

  std::string grayscale_kernel_path = "kernels/grayscale.cl";
  std::string grayscale_kernel_str = readStringFromFile(grayscale_kernel_path);
  cl::Program::Sources grayscale_source(
      1, std::make_pair(grayscale_kernel_str.c_str(),
                        grayscale_kernel_str.size()));
  cl::Program grayscale_program = cl::Program(context, grayscale_source);
  grayscale_program.build(devices);

  cl::Kernel grayscale_kernel(grayscale_program, "grayscale", &err);

  std::string contour_kernel_path = "kernels/contour.cl";
  std::string contour_kernel_str = readStringFromFile(contour_kernel_path);
  cl::Program::Sources contour_source(
      1, std::make_pair(contour_kernel_str.c_str(), contour_kernel_str.size()));
  cl::Program contour_program = cl::Program(context, contour_source);
  contour_program.build(devices);

  cl::Kernel contour_kernel(contour_program, "contour", &err);

  cl::Event event;
  cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);

  double angle = 360;
  double current_angle = 0;
  double step_angle = 1.125;
  int counter = 1;
  int width = 1280, height = 960, channels = 3;
  int crop_width = 885 - 135, crop_height = 960;
  int crop_xoffset = 135, crop_yoffset = 0;
  auto begin = std::chrono::high_resolution_clock::now();

  cl::Buffer buffer_src(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        channels * width * height);
  unsigned char *src_img = (unsigned char *)queue.enqueueMapBuffer(
      buffer_src, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
      channels * width * height);

  cl::Buffer buffer_dst_crop(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                             3 * crop_width * crop_height);

  cl::Buffer buffer_dst_grayscale(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  crop_width * crop_height);

  cl::Buffer buffer_dst_contour(context,
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                crop_height * 3 * sizeof(float));

  float *points_buffer = (float *)queue.enqueueMapBuffer(
      buffer_dst_contour, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
      sizeof(float) * 3 * crop_height);
  while (angle - current_angle > 1e-6) {

    std::vector<std::vector<cv::Point>> contours;
    std::vector<Point> points;
    std::string image_path =
        "/home/rock/git/scan_bench/imgs/" + std::to_string(counter) + ".jpg";

    double next_angle = current_angle + step_angle;

    readJpeg(image_path.c_str(), src_img, width, height);

    crop_kernel.setArg(0, buffer_src);
    crop_kernel.setArg(1, buffer_dst_crop);
    crop_kernel.setArg(2, width);
    crop_kernel.setArg(3, height);
    crop_kernel.setArg(4, crop_width);
    crop_kernel.setArg(5, crop_height);
    crop_kernel.setArg(6, crop_xoffset);
    crop_kernel.setArg(7, crop_yoffset);

    grayscale_kernel.setArg(0, buffer_dst_crop);
    grayscale_kernel.setArg(1, buffer_dst_grayscale);
    grayscale_kernel.setArg(2, crop_width);
    grayscale_kernel.setArg(3, crop_height);

    contour_kernel.setArg(0, buffer_dst_grayscale);
    contour_kernel.setArg(1, buffer_dst_contour);
    contour_kernel.setArg(2, crop_width);
    contour_kernel.setArg(3, crop_height);
    contour_kernel.setArg(4, (float)current_angle);

    cl::Event cropEvent, grayEvent, contourEvent;
    int crop_ndrange = (crop_width * crop_height + 14) / 15;
    auto start_part = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(crop_kernel, cl::NullRange,
                               cl::NDRange(crop_ndrange), cl::NDRange(64), NULL,
                               &cropEvent);
    cropEvent.wait();
    auto end_part = std::chrono::system_clock::now();
    auto elapsed_part = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_part - start_part);
    printf("Part execution time is: %0.3f milliseconds \n",
           elapsed_part / 1000000.0);
    double nanoSeconds =
        double(cropEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
               cropEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    printf("GPU Execution time is: %0.3f milliseconds \n",
           nanoSeconds / 1000000.0);
    queue.enqueueNDRangeKernel(grayscale_kernel, cl::NullRange,
                               cl::NDRange(crop_width * crop_height / 10),
                               cl::NDRange(64), NULL, &grayEvent);
    grayEvent.wait();

    queue.enqueueNDRangeKernel(contour_kernel, cl::NullRange,
                               cl::NDRange(crop_height), cl::NDRange(64), NULL,
                               &contourEvent);
    contourEvent.wait();

    for (int i = 0; i < crop_height; ++i) {
      if (points_buffer[3 * i + 2] > 0) {
        std::string vertex = "v " + std::to_string(points_buffer[3 * i]) + " " +
                             std::to_string(points_buffer[3 * i + 1]) + " " +
                             std::to_string(points_buffer[3 * i + 2]) + "\n";
        result_stream << vertex;
      }
    }

    current_angle = next_angle;
    counter++;
  }
  queue.finish();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

  result_stream.close();
}