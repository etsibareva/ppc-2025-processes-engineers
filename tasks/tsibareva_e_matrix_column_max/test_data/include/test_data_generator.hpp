#pragma once
#include <algorithm>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

inline std::vector<std::vector<int>> GenerateMatrixFunc(int rows, int cols, MatrixType type = MatrixType::ASCENDING) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

  switch (type) {
    case MatrixType::ASCENDING: {
      int counter = 1;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = counter++;
        }
      }
      break;
    }
    case MatrixType::COLUMN_MAX_MIDDLE: {
      int mid_row = rows / 2;
      for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
          if (i == mid_row) {
            matrix[i][j] = 1000000 + j;
          } else {
            matrix[i][j] = (i * cols + j) % 1000;
          }
        }
      }
      break;
    }
    case MatrixType::COLUMN_MAX_RANDOM: {
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
      break;
    }
    case MatrixType::ALL_ZEROS: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = 0;
        }
      }
      break;
    }

    case MatrixType::ALL_NEGATIVE: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = -((i + 1) * 10 + (j + 1));
        }
      }
      break;
    }

    case MatrixType::SINGLE_COLUMN_POSITIVE: {
      int positive_col = cols / 2;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if (j == positive_col) {
            matrix[i][j] = 1000 + i * 10;
          } else {
            matrix[i][j] = -((i + 1) * 10 + (j + 1));
          }
        }
      }
      break;
    }

    case MatrixType::FEW_DENSE_COLUMNS: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if (j < 2) {
            matrix[i][j] = i * 100 + j * 10 + 5;
          } else {
            matrix[i][j] = (i + j) % 3;
          }
        }
      }
      break;
    }

    case MatrixType::IDENTICAL_COLUMNS: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = i * 10 + 5;
        }
      }
      break;
    }

    case MatrixType::IDENTICAL_ROWS: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = j * 10 + 3;
        }
      }
      break;
    }

    case MatrixType::DUPLICATE_MAXUM: {
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
      break;
    }

    case MatrixType::LARGE_VALUES: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = 1000000 + i * 1000 + j;
        }
      }
      break;
    }

    case MatrixType::SMALL_VALUES: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = (i + j) % 5;
        }
      }
      break;
    }

    case MatrixType::DENSE: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = i * cols + j + 1;
        }
      }
      break;
    }

    case MatrixType::DESCENDING: {
      int counter = rows * cols;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = counter--;
        }
      }
      break;
    }

    case MatrixType::CONSTANT: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = 42;
        }
      }
      break;
    }

    case MatrixType::DIAGONAL_DOMINANT: {
      int min_dim = std::min(rows, cols);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if (i == j && i < min_dim) {
            matrix[i][j] = 1000 + i * 100;
          } else {
            matrix[i][j] = i + j;
          }
        }
      }
      break;
    }

    case MatrixType::SPARSE: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if ((i + j) % 3 == 0) {
            matrix[i][j] = i * 10 + j + 1;
          } else {
            matrix[i][j] = 0;
          }
        }
      }
      break;
    }

    case MatrixType::NEGATIVE: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix[i][j] = -((i + 1) * 10 + (j + 1));
        }
      }
      break;
    }

    case MatrixType::COLUMN_MAX_LAST: {
      for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows - 1; ++i) {
          matrix[i][j] = i + j;
        }
        matrix[rows - 1][j] = 1000 + j;
      }
      break;
    }

    case MatrixType::COLUMN_MAX_FIRST: {
      for (int j = 0; j < cols; ++j) {
        matrix[0][j] = 1000 + j;
        for (int i = 1; i < rows; ++i) {
          matrix[i][j] = i + j;
        }
      }
      break;
    }

    case MatrixType::CHECKERBOARD: {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if ((i + j) % 2 == 0) {
            matrix[i][j] = 1;
          } else {
            matrix[i][j] = -1;
          }
        }
      }
      break;
    }
  }

  return matrix;
}

inline std::vector<int> GenerateExpectedOutput(const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty()) {
    return {};
  }

  std::vector<int> result(matrix[0].size(), 0);

  for (size_t j = 0; j < matrix[0].size(); ++j) {
    int max_val = matrix[0][j];
    for (size_t i = 1; i < matrix.size(); ++i) {
      if (matrix[i][j] > max_val) {
        max_val = matrix[i][j];
      }
    }
    result[j] = max_val;
  }
  return result;
}

}  // namespace tsibareva_e_matrix_column_max
