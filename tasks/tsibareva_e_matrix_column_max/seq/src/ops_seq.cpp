#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

TsibarevaEMatrixColumnMaxSEQ::TsibarevaEMatrixColumnMaxSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  const auto &flat_matrix = std::get<0>(in);
  int rows = std::get<1>(in);
  int cols = std::get<2>(in);

  flat_input_ = flat_matrix;
  rows_ = rows;
  cols_ = cols;

  GetOutput() = std::vector<int>();
}

bool TsibarevaEMatrixColumnMaxSEQ::ValidationImpl() {
  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::PreProcessingImpl() {
  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  GetOutput() = std::vector<int>(cols_, 0);
  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

  auto &column_maxs = GetOutput();

  for (int col = 0; col < cols_; ++col) {
    int max_value = flat_input_[static_cast<size_t>(col) * rows_];
    for (int row = 1; row < rows_; ++row) {
      int idx = (col * rows_) + row;
      int element = flat_input_[idx];
      max_value = std::max(element, max_value);
    }
    column_maxs[col] = max_value;
  }

  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_matrix_column_max
