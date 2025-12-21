#include "tsibareva_e_edge_select_sobel/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

const std::vector<std::vector<int>> SOBEL_X = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const std::vector<std::vector<int>> SOBEL_Y = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

TsibarevaEEdgeSelectSobelSEQ::TsibarevaEEdgeSelectSobelSEQ(const InType &in)
    : height_(std::get<1>(in)), width_(std::get<2>(in)), threshold_(std::get<3>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TsibarevaEEdgeSelectSobelSEQ::ValidationImpl() {
  if (height_ == 0 || width_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>(static_cast<size_t>(height_ * width_), 0);
  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::RunImpl() {
  const auto &flat_pixels_ = std::get<0>(GetInput());
  input_pixels_ = std::vector<int>(flat_pixels_);

  auto &output_pixels_ = GetOutput();

  if (height_ < 3 || width_ < 3) {
    std::copy(input_pixels_.begin(), input_pixels_.end(), output_pixels_.begin());
    return true;
  }

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int gx = CalculateGradientX(input_pixels_, x, y);
      int gy = CalculateGradientY(input_pixels_, x, y);

      int m = static_cast<int>(std::sqrt(gx * gx + gy * gy + 0.0));
      int result = (m <= threshold_) ? 0 : m;

      output_pixels_[static_cast<size_t>(y * width_ + x)] = result;
    }
  }

  std::cout << "Result SEQ" << " threshold=" << threshold_ << ":" << std::endl;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      std::cout << GetOutput()[static_cast<size_t>(y * width_ + x)] << " ";
    }
    std::cout << std::endl;
  }

  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::PostProcessingImpl() {
  return true;
}

int TsibarevaEEdgeSelectSobelSEQ::CalculateGradientX(const std::vector<int> &pixels, int x, int y) const {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      int weight = SOBEL_X[ky + 1][kx + 1];

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        sum += weight * pixels[static_cast<size_t>(ny * width_ + nx)];
      }
    }
  }

  return sum;
}

int TsibarevaEEdgeSelectSobelSEQ::CalculateGradientY(const std::vector<int> &pixels, int x, int y) const {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      int weight = SOBEL_Y[ky + 1][kx + 1];

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        sum += weight * pixels[static_cast<size_t>(ny * width_ + nx)];
      }
    }
  }

  return sum;
}

}  // namespace tsibareva_e_edge_select_sobel
