#pragma once

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

  void CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size, size_t num_cols);
  void StoreProcessorResults(int proc, const std::vector<int> &proc_maxs, int world_size, size_t num_cols);
  int CountColumnsForProcessor(int proc, int world_size, size_t num_cols);
  void SendLocalResults(const std::vector<int> &local_maxs);

  std::vector<int> final_result_;
};

}  // namespace tsibareva_e_matrix_column_max
