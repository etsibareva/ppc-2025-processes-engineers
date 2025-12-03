#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

TsibarevaEMatrixColumnMaxMPI::TsibarevaEMatrixColumnMaxMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    flat_input_ = std::get<0>(in);
    rows_ = std::get<1>(in);
    cols_ = std::get<2>(in);
  } else {
    flat_input_ = std::vector<int>();
    rows_ = -1;
    cols_ = -1;
  }
  GetOutput() = std::vector<int>();
}

bool TsibarevaEMatrixColumnMaxMPI::ValidationImpl() {
  return true;
}

bool TsibarevaEMatrixColumnMaxMPI::PreProcessingImpl() {
  return true;
}

bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  // Подготовка начальных значений, начальные проверки.
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  // Расчет количества столбцов на процесс (локально).
  int base_cols = cols_ / world_size;
  int remainder = cols_ % world_size;
  local_cols_ = base_cols + (world_rank < remainder ? 1 : 0);

  // Расчет смещений (на процессе 0 с рассылкой).
  std::vector<int> send_counts(world_size);
  std::vector<int> displacements(world_size);
  if (world_rank == 0) {
    int displ = 0;
    for (int i = 0; i < world_size; i++) {
      int proc_cols = base_cols + (i < remainder ? 1 : 0);
      send_counts[i] = proc_cols * rows_;
      displacements[i] = displ;
      displ += send_counts[i];
    }
  }
  MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылка входных данных (с процесса 0).
  local_flat_data_.resize(local_cols_ * rows_);
  MPI_Scatterv(world_rank == 0 ? flat_input_.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
               local_flat_data_.data(), static_cast<int>(local_flat_data_.size()), MPI_INT, 0, MPI_COMM_WORLD);

  // Подсчет максимумов (локально).
  std::vector<int> local_maxs(local_cols_, std::numeric_limits<int>::min());
  for (int col = 0; col < local_cols_; col++) {
    for (int row = 0; row < rows_; row++) {
      int idx = col * rows_ + row;
      local_maxs[col] = std::max(local_flat_data_[idx], local_maxs[col]);
    }
  }

  // Подготовка буфера приёма и смещений в нём (на каждом процессе, с целью синхронизации результатов GetOutput).
  std::vector<int> recv_counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<int> global_result(cols_);
  int total_displ = 0;
  for (int i = 0; i < world_size; i++) {
    int proc_cols = base_cols + (i < remainder ? 1 : 0);
    recv_counts[i] = proc_cols;
    displs[i] = total_displ;
    total_displ += proc_cols;
  }

  // Сбор и синхронизация результатов.
  MPI_Allgatherv(local_maxs.data(), local_cols_, MPI_INT, global_result.data(), recv_counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  GetOutput() = global_result;

  return true;
}

bool TsibarevaEMatrixColumnMaxMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_matrix_column_max
