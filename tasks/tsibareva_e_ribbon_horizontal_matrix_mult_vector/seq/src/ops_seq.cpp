#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

TsibarevaERibbonHorizontalMatrixMultVectorSEQ::TsibarevaERibbonHorizontalMatrixMultVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  const auto &flat_matrix = std::get<0>(in);
  int rows = std::get<1>(in);
  int cols = std::get<2>(in);
  const auto &flat_vector = std::get<3>(in);

  input_matrix_ = flat_matrix;
  rows_ = rows;
  cols_ = cols;
  input_vector_ = flat_vector;

  GetOutput() = std::vector<int>();
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::ValidationImpl() {
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::PreProcessingImpl() {
  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  GetOutput() = std::vector<int>(rows_, 0);
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::RunImpl() {
  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

  if (std::cmp_not_equal(cols_, static_cast<int>(input_vector_.size()))) {
    return true;
  }

  auto &result_vector = GetOutput();

  for (int row = 0; row < rows_; ++row) {
    int sum = 0;
    for (int col = 0; col < cols_; ++col) {
      int matrix_idx = (row * cols_) + col;
      sum += input_matrix_[static_cast<size_t>(matrix_idx)] * input_vector_[static_cast<size_t>(col)];
    }
    result_vector[row] = sum;
  }

  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
