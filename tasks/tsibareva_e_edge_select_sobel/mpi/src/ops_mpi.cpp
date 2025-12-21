#include "tsibareva_e_edge_select_sobel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

const std::vector<std::vector<int>> SOBEL_X = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const std::vector<std::vector<int>> SOBEL_Y = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

TsibarevaEEdgeSelectSobelMPI::TsibarevaEEdgeSelectSobelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    GetInput() = in;
    input_pixels_ = std::get<0>(GetInput());
    height_ = std::get<1>(GetInput());
    width_ = std::get<2>(GetInput());
    threshold_ = std::get<3>(GetInput());
  }

  GetOutput() = std::vector<int>();
}

bool TsibarevaEEdgeSelectSobelMPI::ValidationImpl() {
  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::PreProcessingImpl() {
  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastParameters();

  DistributeRows();

  std::vector<int> local_result = ComputeLocalGradients();

  GatherResults(local_result);

  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::PostProcessingImpl() {
  return true;
}

void TsibarevaEEdgeSelectSobelMPI::BroadcastParameters() {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&threshold_, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaEEdgeSelectSobelMPI::DistributeRows() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // базовое (основное) количество строк на процесс
  int base_rows = height_ / world_size;
  int remainder = height_ % world_size;

  // предварительное количество строк на процесс
  int real_rows = base_rows + (world_rank < remainder ? 1 : 0);
  local_height_ = real_rows;

  // отдельно подсчитаны флаги, какому процессу требуется верхняя соседняя строка, какому - нижняя соседняя строка
  int need_top_halo = (world_rank > 0) ? 1 : 0;
  int need_bottom_halo = (world_rank < world_size - 1) ? 1 : 0;

  // итоговое количество строк на процесс (и подготовка локального буфера)
  int total_rows = real_rows + need_top_halo + need_bottom_halo;
  local_height_with_halo_ = total_rows;
  local_pixels_.resize(static_cast<size_t>(total_rows) * width_, 0);

  // расчет смещений в исходном массиве
  std::vector<int> send_counts(world_size, 0);
  std::vector<int> send_displs(world_size, 0);
  std::vector<int> real_rows_per_proc(world_size, 0);

  if (world_rank == 0) {
    int current_row = 0;
    for (int dest = 0; dest < world_size; ++dest) {
      int dest_real_rows = base_rows + (dest < remainder ? 1 : 0);
      real_rows_per_proc[dest] = dest_real_rows;

      int dest_need_top_halo = (dest > 0) ? 1 : 0;
      int dest_need_bottom_halo = (dest < world_size - 1) ? 1 : 0;

      int start_row_with_halo = current_row - dest_need_top_halo;
      if (start_row_with_halo < 0) {
        start_row_with_halo = 0;
      }

      int end_row_with_halo = current_row + dest_real_rows + dest_need_bottom_halo - 1;
      if (end_row_with_halo >= height_) {
        end_row_with_halo = height_ - 1;
      }

      int actual_rows = end_row_with_halo - start_row_with_halo + 1;

      send_counts[dest] = actual_rows * width_;
      send_displs[dest] = start_row_with_halo * width_;

      current_row += dest_real_rows;
    }
  }

  MPI_Scatterv(world_rank == 0 ? input_pixels_.data() : nullptr, send_counts.data(), send_displs.data(), MPI_INT,
               local_pixels_.data(), static_cast<int>(local_pixels_.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> TsibarevaEEdgeSelectSobelMPI::ComputeLocalGradients() {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> local_result;
  if (local_height_ > 0) {
    local_result.resize(static_cast<size_t>(local_height_) * width_, 0);

    for (int local_y = 0; local_y < local_height_; ++local_y) {
      int y_in_local_data = local_y + ((world_rank > 0) ? 1 : 0);

      for (int x = 0; x < width_; ++x) {
        int gx = CalculateGradientX(x, y_in_local_data);
        int gy = CalculateGradientY(x, y_in_local_data);

        int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy + 0.0));

        local_result[static_cast<size_t>(local_y * width_ + x)] = (magnitude <= threshold_) ? 0 : magnitude;
      }
    }
  }

  return local_result;
}

int TsibarevaEEdgeSelectSobelMPI::CalculateGradientX(int x, int y_in_local_data) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y_in_local_data + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[static_cast<size_t>(ny * width_ + nx)];
        sum += pixel * SOBEL_X[ky + 1][kx + 1];
      }
    }
  }

  return sum;
}

int TsibarevaEEdgeSelectSobelMPI::CalculateGradientY(int x, int y_in_local_data) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y_in_local_data + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[static_cast<size_t>(ny * width_ + nx)];
        sum += pixel * SOBEL_Y[ky + 1][kx + 1];
      }
    }
  }

  return sum;
}

void TsibarevaEEdgeSelectSobelMPI::GatherResults(const std::vector<int> &local_result) {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> local_result_sizes(world_size);
  int local_size = static_cast<int>(local_result.size());
  MPI_Allgather(&local_size, 1, MPI_INT, local_result_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(world_size);
  int total_size = 0;
  for (int i = 0; i < world_size; ++i) {
    displs[i] = total_size;
    total_size += local_result_sizes[i];
  }

  GetOutput().resize(static_cast<size_t>(total_size));

  MPI_Allgatherv(local_result.empty() ? nullptr : local_result.data(), local_size, MPI_INT, GetOutput().data(),
                 local_result_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
}

}  // namespace tsibareva_e_edge_select_sobel
