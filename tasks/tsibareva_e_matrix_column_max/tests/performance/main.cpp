#include <gtest/gtest.h>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"
#include "tsibareva_e_matrix_column_max/test_data/include/test_data_generator.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixRows_ = 5000;
  const int kMatrixCols_ = 5000;
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    input_data_ = GenerateMatrixFunc(kMatrixRows_, kMatrixCols_, MatrixType::kColumnMaxMiddle);
    expected_output_ = GenerateExpectedOutput(input_data_);
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
