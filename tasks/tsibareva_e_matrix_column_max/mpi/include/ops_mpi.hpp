#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaEMatrixColumnMaxMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TsibarevaEMatrixColumnMaxMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeData();

  std::vector<int> flat_input_;
  std::vector<int> local_flat_data_;
  int rows_ = 0;
  int cols_ = 0;
  int local_cols_ = 0;
};

}  // namespace tsibareva_e_matrix_column_max
