#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tsibareva_e_matrix_column_max {

TsibarevaEMatrixColumnMaxSEQ::TsibarevaEMatrixColumnMaxSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = std::vector<std::vector<int>>(in);
  // GetInput().swap(const_cast<InType &>(in)); (не сработало в CI)
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
  size_t num_cols = matrix[0].size();

  for (size_t col = 0; col < num_cols; ++col) {
    int max_val = matrix[0][col];
    for (size_t row = 1; row < matrix.size(); ++row) {
      if (matrix[row][col] > max_val) {
        max_val = matrix[row][col];
      }
    }
    column_maxs[col] = max_val;
  }

  return true;
}

bool TsibarevaEMatrixColumnMaxSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_matrix_column_max
