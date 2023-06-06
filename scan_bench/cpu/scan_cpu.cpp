#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

bool less_by_y(const cv::Point &lhs, const cv::Point &rhs) {
  return lhs.y > rhs.y;
}

struct Point {
  double x, y, z;
};

const int LEFT = 135;
const int RIGHT = 885;
const double x_size = 514; // 526

int main(int argc, char **argv) {

  std::ofstream result_stream;
  result_stream.open("/home/rock/git/scan_bench/cpu/build/result.obj");

  auto begin = std::chrono::high_resolution_clock::now();
  double angle = 360;
  double current_angle = 0;
  double step_angle = 1.125;
  int counter = 1;

  while (angle - current_angle > 1e-6) {

    cv::Mat src, image;

    src = imread("/home/rock/git/scan_bench/imgs/" + std::to_string(counter) +
                     ".jpg",
                 cv::IMREAD_COLOR);

    double next_angle = current_angle + step_angle;

    image = src(cv::Rect(LEFT, 0, RIGHT - LEFT, src.size().height)).clone();

    cv::Mat gray_image;
    auto start_part = std::chrono::high_resolution_clock::now();
    cv::cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
    cv::Mat white_image;
    cv::inRange(gray_image, cv::Scalar(200, 200, 200),
                cv::Scalar(255, 255, 255), white_image);
    auto end_part = std::chrono::system_clock::now();
    // double sigma = 3.0;
    // double lowThreshold = 0.0;
    // double highThreshold = 0.5 * 255.0;

    // cv::Mat edges;
    // cv::Canny(white_image, edges, lowThreshold, highThreshold, 3, true);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<Point> points;
    // cv::imwrite("white.jpg", white_image);

    cv::findContours(white_image, contours, cv::RETR_TREE,
                     cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

    for (size_t j = 0; j < contours.size(); ++j) {
      std::vector<cv::Point>::iterator result =
          std::min_element(contours[j].begin(), contours[j].end(), less_by_y);
      contours[j].erase(result, contours[j].end());
    }

    for (size_t k = 0; k < contours.size(); ++k) {
      for (size_t m = 0; m < contours[k].size(); ++m) {
        Point point;
        point.x = (contours[k][m].x - x_size) * 2. *
                  cos(current_angle * 3.1415 / 180);
        point.y = (contours[k][m].x - x_size) * 2. *
                  sin(current_angle * 3.1415 / 180);
        point.z = contours[k][m].y;

        points.push_back(point);
      }
    }

    auto elapsed_part = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_part - start_part);
    printf("Part execution time is: %0.3f milliseconds \n",
           elapsed_part / 1000000.0);

    for (size_t i = 0; i < points.size(); ++i) {
      std::string vertex = "v " + std::to_string(points[i].x) + " " +
                           std::to_string(points[i].y) + " " +
                           std::to_string(points[i].z) + "\n";
    }

    current_angle = next_angle;
    counter++;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
  printf("Time measured: %lld nanoseconds.\n", elapsed.count());

  result_stream.close();
}