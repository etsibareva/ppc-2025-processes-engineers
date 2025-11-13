#pragma once

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

  std::vector<int> final_result_;
};

}  // namespace tsibareva_e_matrix_column_max
