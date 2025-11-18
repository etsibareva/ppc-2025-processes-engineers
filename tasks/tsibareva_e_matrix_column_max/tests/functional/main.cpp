#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int rows = std::get<0>(test_param);
    int cols = std::get<1>(test_param);
    std::string desc = std::get<3>(test_param);
    return std::to_string(rows) + "x" + std::to_string(cols) + "_" + desc;
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<0>(params);
    int cols = std::get<1>(params);
    MatrixType matrix_type = std::get<2>(params);
    input_data_ = GenerateMatrixFunc(rows, cols, matrix_type);
    expected_output_ = GenerateExpectedOutput(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(TsibarevaERunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 17> kTestParams = {{std::make_tuple(1, 1, MatrixType::kConstant, "single"),
                                               std::make_tuple(1, 10, MatrixType::kAscending, "single_row"),
                                               std::make_tuple(3, 1, MatrixType::kAscending, "single_col"),

                                               std::make_tuple(5, 5, MatrixType::kAllZeros, "all_zeros"),
                                               std::make_tuple(5, 5, MatrixType::kConstant, "constant"),

                                               std::make_tuple(6, 4, MatrixType::kColumnMaxFirst, "max_first"),
                                               std::make_tuple(6, 4, MatrixType::kColumnMaxLast, "max_last"),
                                               std::make_tuple(6, 4, MatrixType::kColumnMaxMiddle, "max_middle"),

                                               std::make_tuple(8, 8, MatrixType::kAscending, "ascending_simple"),
                                               std::make_tuple(8, 8, MatrixType::kDescending, "descending_simple"),
                                               std::make_tuple(8, 8, MatrixType::kDiagonalDominant, "diagonal_dom"),
                                               std::make_tuple(8, 8, MatrixType::kSparse, "sparse"),
                                               std::make_tuple(8, 8, MatrixType::kNegative, "negative"),

                                               std::make_tuple(2, 2, MatrixType::kAscending, "square_small"),
                                               std::make_tuple(10, 4, MatrixType::kAscending, "vertical"),
                                               std::make_tuple(5, 10, MatrixType::kAscending, "horizontal"),
                                               std::make_tuple(7, 7, MatrixType::kCheckerboard, "checkerboard")}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TsibarevaEMatrixColumnMaxMPI, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_matrix_column_max),
                                           ppc::util::AddFuncTask<TsibarevaEMatrixColumnMaxSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_matrix_column_max));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsibarevaERunFuncTestsProcesses::PrintFuncTestName<TsibarevaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TsibarevaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

TEST(TsibarevaEMatrixColumnMaxMPI, EmptyMatrixShouldFailValidation) {
  auto matrix = GenerateMatrixFunc(0, 0, MatrixType::kAscending);
  TsibarevaEMatrixColumnMaxMPI test_task(matrix);
  bool success = test_task.Validation();
  ASSERT_FALSE(success);
}

TEST(TsibarevaEMatrixColumnMaxSEQ, EmptyMatrixShouldFailValidation) {
  auto matrix = GenerateMatrixFunc(0, 0, MatrixType::kAscending);
  TsibarevaEMatrixColumnMaxSEQ test_task(matrix);
  bool success = test_task.Validation();
  ASSERT_FALSE(success);
}

TEST(TsibarevaEMatrixColumnMaxMPI, ZeroColumnsMatrixShouldFailValidation) {
  auto matrix = GenerateMatrixFunc(5, 0, MatrixType::kAscending);
  TsibarevaEMatrixColumnMaxMPI test_task(matrix);
  bool success = test_task.Validation();
  ASSERT_FALSE(success);
}

TEST(TsibarevaEMatrixColumnMaxSEQ, ZeroColumnsMatrixShouldFailValidation) {
  auto matrix = GenerateMatrixFunc(5, 0, MatrixType::kAscending);
  TsibarevaEMatrixColumnMaxSEQ test_task(matrix);
  bool success = test_task.Validation();
  ASSERT_FALSE(success);
}

}  // namespace

}  // namespace tsibareva_e_matrix_column_max
