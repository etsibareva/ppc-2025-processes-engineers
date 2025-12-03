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

  void BroadcastMatrixDimensions();
  void CalculateLocalColumns(int world_rank, int world_size);
  void PrepareScatterParameters(int world_rank, int world_size, std::vector<int> &send_counts,
                                std::vector<int> &displacements) const;
  void ScatterMatrixData(int world_rank, const std::vector<int> &send_counts, const std::vector<int> &displacements);
  std::vector<int> CalculateLocalColumnMaxima();
  std::vector<int> GatherGlobalResults(int world_size, const std::vector<int> &local_maxs) const;

  std::vector<int> flat_input_;
  std::vector<int> local_flat_data_;
  int rows_ = 0;
  int cols_ = 0;
  int local_cols_ = 0;
};

}  // namespace tsibareva_e_matrix_column_max
