#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace tsibareva_e_matrix_column_max {

enum class MatrixType {
    ASCENDING,               // Возрастающие значения
    DESCENDING,              // Убывающие значения  
    CONSTANT,                // Все одинаковые
    DIAGONAL_DOMINANT,       // Большие значения на диагонали
    SPARSE,                  // Разреженная
    NEGATIVE,                // Только отрицательные
    COLUMN_MAX_LAST,         // Максимум в последней строке
    COLUMN_MAX_FIRST,        // Максимум в первой строке
    COLUMN_MAX_MIDDLE,       // Максимум в середине
    CHECKERBOARD,            // Шахматное рапределение
    DENSE,                   // Плотная
    COLUMN_MAX_RANDOM,       // Максимум в псевдослучайной строке
    ALL_ZEROS,               // Заполнена нулями
    ALL_NEGATIVE,            // Заполнена отрицательными значениями
    SINGLE_COLUMN_POSITIVE,  // Все отрицательные значения, кроме одного столбца
    FEW_DENSE_COLUMNS,       // Первые столбцы платные, остальные разреженные или нулевые
    IDENTICAL_COLUMNS,       // Все столбцы одинаковые
    IDENTICAL_ROWS,          // Все строки одинаковые
    DUPLICATE_MAXUM,         // Дублирующиеся максимумы
    LARGE_VALUES,            // Большие числа
    SMALL_VALUES             // Маленькие числа
};

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsibareva_e_matrix_column_max
 