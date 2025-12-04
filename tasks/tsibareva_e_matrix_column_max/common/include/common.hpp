#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_matrix_column_max {

enum class MatrixType : std::uint8_t {
  kSingleConstant,    // 1x1 константная
  kSingleRow,         // 1x10 одна строка
  kSingleCol,         // 3x1 один столбец
  kAllZeros,          // 5x5 все нули
  kConstant,          // 5x5 константная
  kMaxFirst,          // 6x4 максимум в первой строке
  kMaxLast,           // 6x4 максимум в последней строке
  kMaxMiddle,         // 6x4 максимум в середине
  kAscending,         // 8x8 возрастающая
  kDescending,        // 8x8 убывающая
  kDiagonalDominant,  // 8x8 диагонально доминантная
  kSparse,            // 8x8 разреженная
  kNegative,          // 8x8 отрицательная
  kSquareSmall,       // 2x2 маленькая квадратная
  kVertical,          // 10x4 вертикальная
  kHorizontal,        // 5x10 горизонтальная
  kCheckerboard,      // 7x7 шахматная
  kEmpty,             // Пустая матрица
  kZeroColumns        // Матрица с нулевыми столбцами
};

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::tuple<std::vector<int>, int, int> GenerateSingleConstantMatrix() {
  return {{30}, 1, 1};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {30};
}

inline std::tuple<std::vector<int>, int, int> GenerateSingleRowMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1, 10};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
}

inline std::tuple<std::vector<int>, int, int> GenerateSingleColMatrix() {
  return {{1, 2, 3}, 3, 1};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3};
}

inline std::tuple<std::vector<int>, int, int> GenerateAllZerosMatrix() {
  std::vector<int> matrix(25, 0);
  return {matrix, 5, 5};
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0, 0};
}

inline std::tuple<std::vector<int>, int, int> GenerateConstantMatrix() {
  std::vector<int> matrix(25, 30);
  return {matrix, 5, 5};
}
inline std::vector<int> GenerateConstantExpected() {
  return {30, 30, 30, 30, 30};
}

inline std::tuple<std::vector<int>, int, int> GenerateMaxFirstMatrix() {
  std::vector<int> matrix = {1000, 1, 2, 3, 4, 5, 1001, 2, 3, 4, 5, 6, 1002, 3, 4, 5, 6, 7, 1003, 4, 5, 6, 7, 8};
  return {matrix, 6, 4};
}
inline std::vector<int> GenerateMaxFirstExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::tuple<std::vector<int>, int, int> GenerateMaxLastMatrix() {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 1000, 2, 3, 4, 5, 6, 1001, 3, 4, 5, 6, 7, 1002, 4, 5, 6, 7, 8, 1003};
  return {matrix, 6, 4};
}
inline std::vector<int> GenerateMaxLastExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::tuple<std::vector<int>, int, int> GenerateMaxMiddleMatrix() {
  std::vector<int> matrix = {1, 2, 1000, 4, 5, 6, 2, 3, 1001, 5, 6, 7, 3, 4, 1002, 6, 7, 8, 4, 5, 1003, 7, 8, 9};
  return {matrix, 6, 4};
}
inline std::vector<int> GenerateMaxMiddleExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::tuple<std::vector<int>, int, int> GenerateAscendingMatrix() {
  std::vector<int> matrix = {1,  9,  17, 25, 33, 41, 49, 57, 2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43,
                             51, 59, 4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61, 6,  14, 22, 30,
                             38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63, 8,  16, 24, 32, 40, 48, 56, 64};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}

inline std::tuple<std::vector<int>, int, int> GenerateDescendingMatrix() {
  std::vector<int> matrix = {64, 56, 48, 40, 32, 24, 16, 8,  63, 55, 47, 39, 31, 23, 15, 7,  62, 54, 46, 38, 30, 22,
                             14, 6,  61, 53, 45, 37, 29, 21, 13, 5,  60, 52, 44, 36, 28, 20, 12, 4,  59, 51, 43, 35,
                             27, 19, 11, 3,  58, 50, 42, 34, 26, 18, 10, 2,  57, 49, 41, 33, 25, 17, 9,  1};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateDescendingExpected() {
  return {64, 63, 62, 61, 60, 59, 58, 57};
}

inline std::tuple<std::vector<int>, int, int> GenerateDiagonalDominantMatrix() {
  std::vector<int> matrix = {1000, 1, 2,    3, 4,    5,  6,    7,  1, 1100, 3, 4,    5,  6,    7,  8,
                             2,    3, 1200, 5, 6,    7,  8,    9,  3, 4,    5, 1300, 7,  8,    9,  10,
                             4,    5, 6,    7, 1400, 9,  10,   11, 5, 6,    7, 8,    9,  1500, 11, 12,
                             6,    7, 8,    9, 10,   11, 1600, 14, 7, 8,    9, 10,   11, 12,   14, 1700};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateDiagonalDominantExpected() {
  return {1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700};
}

inline std::tuple<std::vector<int>, int, int> GenerateSparseMatrix() {
  std::vector<int> matrix = {8, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 61, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0,  0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateSparseExpected() {
  return {8, 9, 0, 61, 0, 0, 72, 0};
}

inline std::tuple<std::vector<int>, int, int> GenerateNegativeMatrix() {
  std::vector<int> matrix = {-30, -40, -50, -60, -70, -80, -90, -100, -31, -41, -51, -61, -71, -81, -91, -101,
                             -32, -42, -52, -62, -72, -82, -92, -102, -33, -43, -53, -63, -73, -83, -93, -103,
                             -34, -44, -54, -64, -74, -84, -94, -104, -35, -45, -55, -65, -75, -85, -95, -105,
                             -36, -46, -56, -66, -76, -86, -96, -106, -37, -47, -57, -67, -77, -87, -97, -107};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateNegativeExpected() {
  return {-30, -31, -32, -33, -34, -35, -36, -37};
}

inline std::tuple<std::vector<int>, int, int> GenerateSquareSmallMatrix() {
  std::vector<int> matrix = {1, 3, 2, 4};
  return {matrix, 2, 2};
}
inline std::vector<int> GenerateSquareSmallExpected() {
  return {3, 4};
}

inline std::tuple<std::vector<int>, int, int> GenerateVerticalMatrix() {
  std::vector<int> matrix = {1, 5, 9,  13, 17, 21, 25, 29, 33, 37, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38,
                             3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40};
  return {matrix, 10, 4};
}
inline std::vector<int> GenerateVerticalExpected() {
  return {37, 38, 39, 40};
}

inline std::tuple<std::vector<int>, int, int> GenerateHorizontalMatrix() {
  std::vector<int> matrix = {1,  11, 21, 31, 41, 2,  12, 22, 32, 42, 3,  13, 23, 33, 43, 4,  14,
                             24, 34, 44, 5,  15, 25, 35, 45, 6,  16, 26, 36, 46, 7,  17, 27, 37,
                             47, 8,  18, 28, 38, 48, 9,  19, 29, 39, 49, 10, 20, 30, 40, 50};
  return {matrix, 5, 10};
}
inline std::vector<int> GenerateHorizontalExpected() {
  return {41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
}

inline std::tuple<std::vector<int>, int, int> GenerateCheckerboardMatrix() {
  std::vector<int> matrix = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
                             1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};
  return {matrix, 7, 7};
}
inline std::vector<int> GenerateCheckerboardExpected() {
  return {1, 1, 1, 1, 1, 1, 1};
}

inline std::tuple<std::vector<int>, int, int> GenerateEmptyMatrix() {
  return {{}, 0, 0};
}

inline std::tuple<std::vector<int>, int, int> GenerateZeroColumnsMatrix() {
  return {{}, 5, 0};
}

inline std::vector<int> GenerateEmptyExpected() {
  return {};
}

inline std::tuple<std::vector<int>, int, int> GenerateMatrixFunc(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantMatrix();
    case MatrixType::kSingleRow:
      return GenerateSingleRowMatrix();
    case MatrixType::kSingleCol:
      return GenerateSingleColMatrix();
    case MatrixType::kAllZeros:
      return GenerateAllZerosMatrix();
    case MatrixType::kConstant:
      return GenerateConstantMatrix();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstMatrix();
    case MatrixType::kMaxLast:
      return GenerateMaxLastMatrix();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleMatrix();
    case MatrixType::kAscending:
      return GenerateAscendingMatrix();
    case MatrixType::kDescending:
      return GenerateDescendingMatrix();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantMatrix();
    case MatrixType::kSparse:
      return GenerateSparseMatrix();
    case MatrixType::kNegative:
      return GenerateNegativeMatrix();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallMatrix();
    case MatrixType::kVertical:
      return GenerateVerticalMatrix();
    case MatrixType::kHorizontal:
      return GenerateHorizontalMatrix();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardMatrix();
    case MatrixType::kEmpty:
      return GenerateEmptyMatrix();
    case MatrixType::kZeroColumns:
      return GenerateZeroColumnsMatrix();
    default:
      return GenerateSingleConstantMatrix();
  }
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:
      return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:
      return GenerateSingleColExpected();
    case MatrixType::kAllZeros:
      return GenerateAllZerosExpected();
    case MatrixType::kConstant:
      return GenerateConstantExpected();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstExpected();
    case MatrixType::kMaxLast:
      return GenerateMaxLastExpected();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleExpected();
    case MatrixType::kAscending:
      return GenerateAscendingExpected();
    case MatrixType::kDescending:
      return GenerateDescendingExpected();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantExpected();
    case MatrixType::kSparse:
      return GenerateSparseExpected();
    case MatrixType::kNegative:
      return GenerateNegativeExpected();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallExpected();
    case MatrixType::kVertical:
      return GenerateVerticalExpected();
    case MatrixType::kHorizontal:
      return GenerateHorizontalExpected();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardExpected();
    case MatrixType::kEmpty:
    case MatrixType::kZeroColumns:
      return GenerateEmptyExpected();
    default:
      return GenerateSingleConstantExpected();
  }
}

}  // namespace tsibareva_e_matrix_column_max
