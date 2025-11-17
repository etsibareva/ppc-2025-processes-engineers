#pragma once

#include <algorithm>
#include <cstddef>
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

inline std::vector<std::vector<int>> GenerateAscendingMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  int counter = 1;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = counter++;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateDescendingMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  int counter = rows * cols;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = counter--;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateConstantMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = 42;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateAllZerosMatrix(int rows, int cols) {
  return std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0));
}

inline std::vector<std::vector<int>> GenerateAllNegativeMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = -((((i + 1) * 10) + (j + 1)));
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateDenseMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = ((i * cols) + j + 1);
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateSmallValuesMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = (i + j) % 5;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateLargeValuesMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = 1000000 + ((i * 1000) + j);
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateDiagonalDominantMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  int min_dim = std::min(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (i == j && i < min_dim) {
        matrix[i][j] = 1000 + (i * 100);
      } else {
        matrix[i][j] = i + j;
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateSparseMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if ((i + j) % 3 == 0) {
        matrix[i][j] = ((i * 10) + j + 1);
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateNegativeMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = -((((i + 1) * 10) + (j + 1)));
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateCheckerboardMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = ((i + j) % 2 == 0) ? 1 : -1;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateColumnMaxMiddleMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  int mid_row = rows / 2;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (i == mid_row) {
        matrix[i][j] = 1000000 + j;
      } else {
        matrix[i][j] = ((i * cols) + j) % 1000;
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateColumnMaxRandomMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int j = 0; j < cols; ++j) {
    int max_row = j % rows;
    for (int i = 0; i < rows; ++i) {
      if (i == max_row) {
        matrix[i][j] = 1000;
      } else {
        matrix[i][j] = i + j;
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateColumnMaxLastMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows - 1; ++i) {
      matrix[i][j] = i + j;
    }
    matrix[rows - 1][j] = 1000 + j;
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateColumnMaxFirstMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int j = 0; j < cols; ++j) {
    matrix[0][j] = 1000 + j;
    for (int i = 1; i < rows; ++i) {
      matrix[i][j] = i + j;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateSingleColumnPositiveMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  int positive_col = cols / 2;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (j == positive_col) {
        matrix[i][j] = 1000 + (i * 10);
      } else {
        matrix[i][j] = -((((i + 1) * 10) + (j + 1)));
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateFewDenseColumnsMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (j < 2) {
        matrix[i][j] = ((i * 100) + (j * 10) + 5);
      } else {
        matrix[i][j] = (i + j) % 3;
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateIdenticalColumnsMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = (i * 10) + 5;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateIdenticalRowsMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = (j * 10) + 3;
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateDuplicateMaximumMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  for (int j = 0; j < cols; ++j) {
    int max_val = 100 + j;
    for (int i = 0; i < rows; ++i) {
      if (i == rows - 1 || i == rows - 2) {
        matrix[i][j] = max_val;
      } else {
        matrix[i][j] = i + j;
      }
    }
  }
  return matrix;
}

inline std::vector<std::vector<int>> GenerateMatrixFunc(int rows, int cols, MatrixType type = MatrixType::kAscending) {
  switch (type) {
    case MatrixType::kAscending:
      return GenerateAscendingMatrix(rows, cols);
    case MatrixType::kDescending:
      return GenerateDescendingMatrix(rows, cols);
    case MatrixType::kConstant:
      return GenerateConstantMatrix(rows, cols);
    case MatrixType::kAllZeros:
      return GenerateAllZerosMatrix(rows, cols);
    case MatrixType::kAllNegative:
      return GenerateAllNegativeMatrix(rows, cols);
    case MatrixType::kDense:
      return GenerateDenseMatrix(rows, cols);
    case MatrixType::kSmallValues:
      return GenerateSmallValuesMatrix(rows, cols);
    case MatrixType::kLargeValues:
      return GenerateLargeValuesMatrix(rows, cols);
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantMatrix(rows, cols);
    case MatrixType::kSparse:
      return GenerateSparseMatrix(rows, cols);
    case MatrixType::kNegative:
      return GenerateNegativeMatrix(rows, cols);
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardMatrix(rows, cols);
    case MatrixType::kColumnMaxMiddle:
      return GenerateColumnMaxMiddleMatrix(rows, cols);
    case MatrixType::kColumnMaxRandom:
      return GenerateColumnMaxRandomMatrix(rows, cols);
    case MatrixType::kColumnMaxLast:
      return GenerateColumnMaxLastMatrix(rows, cols);
    case MatrixType::kColumnMaxFirst:
      return GenerateColumnMaxFirstMatrix(rows, cols);
    case MatrixType::kSingleColumnPositive:
      return GenerateSingleColumnPositiveMatrix(rows, cols);
    case MatrixType::kFewDenseColumns:
      return GenerateFewDenseColumnsMatrix(rows, cols);
    case MatrixType::kIdenticalColumns:
      return GenerateIdenticalColumnsMatrix(rows, cols);
    case MatrixType::kIdenticalRows:
      return GenerateIdenticalRowsMatrix(rows, cols);
    case MatrixType::kDuplicateMaximum:
      return GenerateDuplicateMaximumMatrix(rows, cols);
  }
  return GenerateAscendingMatrix(rows, cols);
}

inline std::vector<int> GenerateExpectedOutput(const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty()) {
    return {};
  }

  std::vector<int> result(matrix[0].size(), 0);

  for (size_t j = 0; j < matrix[0].size(); ++j) {
    int max_val = matrix[0][j];
    for (size_t i = 1; i < matrix.size(); ++i) {
      max_val = std::max(matrix[i][j], max_val);
    }
    result[j] = max_val;
  }
  return result;
}
}  // namespace tsibareva_e_matrix_column_max
