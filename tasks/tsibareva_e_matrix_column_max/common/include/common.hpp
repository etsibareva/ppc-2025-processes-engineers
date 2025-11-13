#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_matrix_column_max {

enum class MatrixType : std::uint8_t {
  kAscending,             // Возрастающие значения
  kDescending,            // Убывающие значения
  kConstant,              // Все одинаковые
  kDiagonalDominant,      // Большие значения на диагонали
  kSparse,                // Разреженная
  kNegative,              // Только отрицательные
  kColumnMaxLast,         // Максимум в последней строке
  kColumnMaxFirst,        // Максимум в первой строке
  kColumnMaxMiddle,       // Максимум в середине
  kCheckerboard,          // Шахматное рапределение
  kDense,                 // Плотная
  kColumnMaxRandom,       // Максимум в псевдослучайной строке
  kAllZeros,              // Заполнена нулями
  kAllNegative,           // Заполнена отрицательными значениями
  kSingleColumnPositive,  // Все отрицательные значения, кроме одного столбца
  kFewDenseColumns,       // Первые столбцы плотные, остальные разреженные или нулевые
  kIdenticalColumns,      // Все столбцы одинаковые
  kIdenticalRows,         // Все строки одинаковые
  kDuplicateMaximum,      // Дублирующиеся максимумы
  kLargeValues,           // Большие числа
  kSmallValues            // Маленькие числа
};

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsibareva_e_matrix_column_max
