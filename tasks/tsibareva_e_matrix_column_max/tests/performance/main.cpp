#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixRows_ = 20000;
  const int kMatrixCols_ = 20000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    std::vector<int> flat_matrix(static_cast<size_t>(kMatrixRows_) * kMatrixCols_);
    expected_output_.resize(kMatrixCols_, std::numeric_limits<int>::min());

    int row_middle = kMatrixRows_ / 2;
    for (int col = 0; col < kMatrixCols_; ++col) {
      for (int row = 0; row < kMatrixRows_; ++row) {
        int generate_value = 0;
        if (row == row_middle) {
          generate_value = 1000000 + col;
        } else {
          generate_value = ((row * kMatrixCols_) + col) % 1000;
        }

        int idx = (col * kMatrixRows_) + row;
        flat_matrix[idx] = generate_value;

        expected_output_[col] = std::max(generate_value, expected_output_[col]);
      }
    }

    input_data_ = std::make_tuple(flat_matrix, kMatrixRows_, kMatrixCols_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TsibarevaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TsibarevaEMatrixColumnMaxMPI, TsibarevaEMatrixColumnMaxSEQ>(
        PPC_SETTINGS_tsibareva_e_matrix_column_max);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsibarevaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsibarevaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace tsibareva_e_matrix_column_max
