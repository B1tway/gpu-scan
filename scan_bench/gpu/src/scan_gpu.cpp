#define __CL_ENABLE_EXCEPTIONS

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <vuh/array.hpp>
#include <vuh/vuh.h>

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

struct Point {
  double x, y, z;
};

const int LEFT = 135;
const int RIGHT = 885;
const double x_size = 514; // 526

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

struct Params_crop {
  int width;
  int heigth;
  int x_offset;
  int y_offset;
};
struct Params_grayscale {
  int width;
  int heigth;
};
struct Params_contour {
  int width;
  int heigth;
  float angle;
};

int main(void) {

  std::ofstream result_stream;
  result_stream.open("/home/rock/git/scan_bench/gpu/build/result_gpu.obj");

  auto instance = vuh::Instance();
  auto device = instance.devices().at(0);

  double angle = 360;
  double current_angle = 0;
  double step_angle = 1.125;
  int counter = 1;
  int width = 1280, height = 960, channels = 3;
  int crop_width = 885 - 135, crop_height = 960;
  int crop_xoffset = 135, crop_yoffset = 0;
  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<unsigned char> src_img(3 * width * height);
  std::vector<unsigned char> crop_img(3 * crop_width * crop_height);
  std::vector<unsigned char> gray_img(crop_width * crop_height);
  std::vector<float> slice(crop_height * 3);
  using Specs = vuh::typelist<uint32_t>;

  auto crop = vuh::Program<Specs, Params_crop>(device, "spv/crop.spv");
  auto grayscale =
      vuh::Program<Specs, Params_grayscale>(device, "spv/grayscale.spv");
  auto contour = vuh::Program<Specs, Params_contour>(device, "spv/contour.spv");

  while (angle - current_angle > 1e-6) {

    std::string image_path =
        "/home/rock/git/scan_bench/imgs/" + std::to_string(counter) + ".jpg";

    double next_angle = current_angle + step_angle;

    readJpeg(image_path.c_str(), &src_img[0], width, height);

    auto src_vuh = vuh::Array<float>(device, src_img);
    auto crop_vuh = vuh::Array<float>(device, crop_img);
    auto grayscale_vuh = vuh::Array<float>(device, gray_img);
    auto slice_vuh = vuh::Array<float>(device, slice);

    crop.grid((crop_width * crop_height + 14) / 15)
        .spec(64)({width, height, crop_xoffset, crop_yoffset}, src_vuh,
                  crop_vuh);
    grayscale.grid(crop_width * crop_height / 10)
        .spec(64)({crop_width, crop_height}, crop_vuh, grayscale_vuh);
    contour.grid(crop_height)
        .spec(64)({crop_width, crop_height, current_angle}, grayscale_vuh,
                  slice_vuh);

    for (int i = 0; i < crop_height; ++i) {
      if (slice[3 * i + 2] > 0) {
        std::string vertex = "v " + std::to_string(slice[3 * i]) + " " +
                             std::to_string(slice[3 * i + 1]) + " " +
                             std::to_string(slice[3 * i + 2]) + "\n";
        result_stream << vertex;
      }
    }
    current_angle = next_angle;
    counter++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

  result_stream.close();
}