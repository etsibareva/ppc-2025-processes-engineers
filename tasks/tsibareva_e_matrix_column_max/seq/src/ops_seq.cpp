#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

TsibarevaEMatrixColumnMaxSEQ::TsibarevaEMatrixColumnMaxSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = std::vector<std::vector<int>>(in);
  GetOutput() = std::vector<int>();
}

bool TsibarevaEMatrixColumnMaxSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  if (matrix[0].empty()) {
    return false;
  }

  size_t first_row_size = matrix[0].size();
  for (size_t i = 1; i < matrix.size(); ++i) {
    if (matrix[i].size() != first_row_size) {
      return false;
    }
  }

  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::PreProcessingImpl() {
  if (!GetInput().empty()) {
    GetOutput() = std::vector<int>(GetInput()[0].size(), 0);
  }
  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  auto &column_maxs = GetOutput();
  size_t cols_count = matrix[0].size();

  for (size_t col = 0; col < cols_count; ++col) {
    int max_value = matrix[0][col];
    for (size_t row = 1; row < matrix.size(); ++row) {
      max_value = std::max(matrix[row][col], max_value);
    }
    column_maxs[col] = max_value;
  }

  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_matrix_column_max
