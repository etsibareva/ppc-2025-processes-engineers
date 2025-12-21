#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"
#include "tsibareva_e_edge_select_sobel/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_edge_select_sobel/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsibareva_e_edge_select_sobel {

class TsibarevaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string description = std::get<1>(test_param);
    return description;
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    ImageType image_type = std::get<0>(params);
    input_data_ = GenerateTestData(image_type);
    expected_output_ = GenerateExpectedOutput(image_type);

    // Отладочная информация
    auto &input_pixels = std::get<0>(input_data_);
    int height = std::get<1>(input_data_);
    int width = std::get<2>(input_data_);
    int threshold = std::get<3>(input_data_);

    std::cout << "Input image (" << height << "x" << width << "), threshold=" << threshold << ":" << std::endl;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        std::cout << input_pixels[static_cast<size_t>(y * width + x)] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "Expected output (" << height << "x" << width << "), threshold=" << threshold << ":" << std::endl;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        std::cout << expected_output_[static_cast<size_t>(y * width + x)] << " ";
      }
      std::cout << std::endl;
    };
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

const std::array<TestType, 2> kTestParams = {std::make_tuple(ImageType::kTest1, "test1_small_square"),
                                              std::make_tuple(ImageType::kTest2, "test2_real_img")/*,
                                              std::make_tuple(ImageType::kTest3, "test3_large_square"),
                                              std::make_tuple(ImageType::kTest4, "test4_rectangular"),
                                              std::make_tuple(ImageType::kTest5, "test5_pattern")*/};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TsibarevaEEdgeSelectSobelMPI, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_edge_select_sobel),
                                           ppc::util::AddFuncTask<TsibarevaEEdgeSelectSobelSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_edge_select_sobel));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsibarevaERunFuncTestsProcesses::PrintFuncTestName<TsibarevaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TsibarevaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsibareva_e_edge_select_sobel
