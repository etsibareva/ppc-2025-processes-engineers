#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

class TsibarevaEEdgeSelectSobelMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TsibarevaEEdgeSelectSobelMPI(const InType &in);

 private:
  int height_ = 0;
  int width_ = 0;
  int threshold_ = 0;

  int world_rank_ = 0;
  int world_size_ = 0;

  std::vector<int> input_pixels_;

  std::vector<int> local_pixels_;
  int local_height_ = 0;
  int local_height_with_halo_ = 0;

  // Вспомогательные методы
  void BroadcastParameters();
  void DistributeRows();
  std::vector<int> ComputeLocalGradients();
  int CalculateGradientX(int x, int y_in_local_data);
  int CalculateGradientY(int x, int y_in_local_data);
  void GatherResults(const std::vector<int> &local_result);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsibareva_e_edge_select_sobel
