# Нахождение максимальных значений по столбцам матрицы

- Студент: Цибарева Екатерина Алексеевна, группа 3823Б1ПР1
- Технология: SEQ | MPI
- Вариант: 16

## 1. Введение
Задача поиска максимумов по столбцам матрицы критически важна для анализа данных в машинном обучении, финансовом моделировании и научных вычислениях. При работе с матрицами большого размера последовательные алгоритмы показывают результаты, критически сказывающиеся на общей производительности вычислений.
Целью данной работы является реализация и сравнительный анализ последовательной и параллельной версий алгоритма для нахождения максимальных значений и их позиций в каждом столбце матрицы произвольной размерности.

## 2. Постановка задачи
Описание задачи: для каждого столбца заданной матрицы найти значение максимального элемента.
Входной тип данных: вектор векторов значений типа int.
Выходной тип данных: вектор значений типа int.
Ограничения: 
- матрица может состоять из повторяющихся значений либо содержать повторяющиеся значения;
- результирующие вектора последовательной и параллельной реализаций алгоритма не должны различаться;
- для реализации параллеьного алгоритма должен быть использован MPI;
- матрица может быть квадратной или прямоугольной;
- матрица может быть подана на вход алгоритма в не корректном формате, а именно иметь строки различной длины или быть пустой.

## 3. Описание алгоритма (Последовательный)
Основная идея: алгоритм выполняет поиск максимальных значений для каждого столбца матрицы путем последовательного обхода всех элементов. Необходимо для каждого столбца независимо найти наибольший элемент, просматривая все строки этого столбца.

Конвейер выполнения последовательной версии алгоритма предполагает обязательную проверку на не пустоту матрицы и не пустоту её столбцов, код проверки можно видеть ниже.
``` cpp
if (matrix.empty()) {
    return false;
  }
if (matrix[0].empty()) {
    return false;
}
А также - обязательную проверку на равенство длин строк матрицы. 

Шаги алгоритма включают: 
1. инициализацию матрицы:
``` cpp
const auto &matrix = GetInput();
2. обязательную проверку на пустоту поступившей матрицы, если входной вектор пуст - выполнение прекращается с возвращением false;
3. инициализацию выходного вектора:
``` cpp
auto &column_maxs = GetOutput();
4. а также вычисление количества столбцов для дальнейшего расчета:
``` cpp
size_t cols_count = matrix[0].size();
5. алгоритм предполагает последовательный обход всех элементов с квадратичной (m*n, где m и n - количество строк и столбцов соответственно) сложностью. В качестве первоначально присваиваемого максимального значения выбран первый элемент столбца в цикле по стобцам.

Код базового алгоритма поиска максимумов матрицы по столбцам можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  auto &column_maxs = GetOutput();
  size_t cols_count = matrix[0].size();

  for (size_t col = 0; col < cols_count; ++col) {
    int max_value = matrix[0][col];
    for (size_t row = 1; row < matrix.size(); ++row) {
      max_value = std::max(matrix[row][col], max_value);
    }
    column_maxs[col] = max_value;
  }

  return true;
}

## 4. Схема распараллеливания
Модель распределения: циклическое распределение столбцов. Предположена обработка каждым процессом столбцов с шагом, равным world_size. Таким образом, процесс с рангом rank должен обрабатывать столбцы с номерами rank, rank + world_size, rank + 2*world_size и так далее.

При этом процесс 0 выступает в роли координатора:
- выполняет локальные вычисления;
- собирает частичные результаты с процессов рангом 1 и выше;
- формирует результирующий вектор в соответствии с моделью рапределения столбцов между процессами;
- рассылает финальный результат.

В свою очередь, процессы рангом 1 и выше предполагают:
- выполнение локальных вычислений максимумов для назначенных им столбцов;
- передачу результатов вычислений процессу с рангом 0.

Коммуникационная схема: предположена обработка исходных данных в несколько этапов.
Этап 0 предполагает вычисление номеров столбцов каждого процесса.
Этап 1 содержит вычисление каждым процессом локально максимумов по назначенным ему столбцам.
Этап 2 предполагает сбор результата:
- процессы с рангом 1 и выше направляют результаты вычислений процессу 0 с помощью MPI_Send;
- процесс с рангом 0 принимает результаты через MPI_Recv и формирует результирующий вектор;
Этап 3 содержит рассылку финального результата процессом 0 между остальными процессами при помощи MPI_BCast, что гарантирует идентичность выходных данных на всех процессах и позволяет более удобно и корректно проследить успешное прохождение всех запущенных процессов. 

Особенности модели распределения и коммуникационной схемы:
- циклическое распределение столбцов позволяет не только балансировать нагрузку, равномерно распределяя столбцы между процессами, но и вводить логическое условие, универсально ограничивающее обработку матрицы меньшего количества столбцов, чем запущенных процессов; в таком случае предположено не выделение процессу с рангом, большим или равным количеству столбцов матрицы, никаких данных для обработки;
- была предположена минимизация передаваемых данных в пользу реализации алгоритма, пересылающего только значения максимумов с циклическим восстановлением последовательности результирующего вектора, но не пересылка номеров максимумов в векторе;
- были обработаны основные граничные случаи для матриц, содержащих как больше, так и меньше столбцов, чем запущенных для обработки процессов.

Примерный код реализации алгоритма параллельной обработки можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  // <Объявление локальных переменных и вычисление world_rank, world_size>
  
  // Локальные вычисления каждым процессом назначенных ему столбцов
  for (auto col = static_cast<size_t>(world_rank); col < cols_count; col += static_cast<size_t>(world_size)) {
    int max_val = matrix[0][col];
    for (size_t row = 1; row < rows_count; ++row) {
      max_val = std::max(matrix[row][col], max_val);
    }
    local_maxs.push_back(max_val);
  }

  if (world_rank == 0) {
    // Получение частичных результатов и формирование результирующего вектора на процессе 0
    CollectResultsFromAllProcesses(local_maxs, world_size, cols_count);
  } else {
    // Рассылка результатов процессами рангом 1 и выше
    MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

``` cpp
void TsibarevaEMatrixColumnMaxMPI::CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size, size_t cols_count) {
  // <Объявление локальных переменных и вычисление world_rank, world_size>

  // Копирование локальных результатов процесса 0
  size_t idx = 0;
  for (size_t col = 0; col < cols_count && idx < local_maxs.size(); col += world_size) {
    final_result_[col] = local_maxs[idx++];
  }

  // Получение и копирование локальных результатов процессов рангом 1 и выше
  for (int proc = 1; proc < world_size; proc++) {
    // <Вычисление количества вычисленных процессом значений>

    // Получение результата
    std::vector<int> proc_maxs(static_cast<size_t>(proc_pass));
    MPI_Recv(proc_maxs.data(), proc_pass, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Копирование результата
    size_t proc_idx = 0;
    for (size_t col = proc; col < cols_count; col += world_size) {
      final_result_[col] = proc_maxs[proc_idx++];
    }
  }
} 

## 5. Детали реализации
Проект реализации алгоритмов имеет следующую структуру.
1) Последовательная реализация (seq):
1.1) ops_seq.hpp - объявление класса TsibarevaEMatrixColumnMaxSEQ, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl());
1.2) ops_seq.cpp - реализация перегруженных методов: 
- RunImpl() - реализован базовый алгоритм поиска максимумов;
- PreProcessingImpl() - инициализирован выходной вектор;
- ValidationImpl() - проверены случаи не корректного задания матрицы (пустота, неравенство длин строк);
- PostProcessingImpl() - не предположено отдельной логики.

2) MPI реализация (mpi):
2.1) ops_mpi.hpp - объявление класса TsibarevaEMatrixColumnMaxMPI, содержащего аналогичные методы, обязательные к переопределению, отмеченные как "переопределен" в пункте 2.2 настоящего перечисления;
2.2) ops_mpi.cpp - реализация методов:
- RunImpl() - (переопределен) параллельный алгоритм поиска максимумов, распределяющий содержательные действия между процессами;
- CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size, size_t cols_count) - метод, используемый только процессом 0, осуществляющим координирование вычислений между процессами; предполагает сбор частичных результатов и формирование результирующего вектора;
- PostProcessingImpl() - (переопределен) алгоритм, имеющий назначение рассылки финального результата между процессами в конце выполнения обработки;

Необходимо отметить, что для координирования действий между функциями реализации параллельного алгоритма была введена глобальная переменная std::vector<int> final_result_, объявленная в файле ops_mpi.hpp.

3) Общие компоненты (common):
3.1) common.hpp - содержит:
- объявление входного (InType), выходного (OutType) типа, типа базовой задачи (BaseTask) и тестовых классов (TestTask); 
- также в файле было размещено перечисление типов матриц, доступных для генерации (enum MatrixType);
- функции генерации матриц 17 типов в соответствии с перечислением;
- функцию генерации матрицы GenerateMatrixFunc, в качестве аргументов принимающую количество строк, столбцов и тип генерируемой матрицы, и принимающей решение о вызове соответствующей функции генерации матрицы;
- функцию GenerateExpectedOutput, генерирующую вектор ожидаемых значений на основании поступившей матрицы; при этом предположен последовательный алгоритм вычисления максимумов, тем не менее не влияющий на производительность значительно, поскольку тестирование с использованием генерации и вычисления ожидаемых данных предположено на небольших размерах матриц и имеет целью проверку покрытия всех случаев вычисления максимумов по столбцам.

Необходимо отметить, что в проекте предположена генерация тестовых матриц следующих типов. Примерный код перечисления можно видеть ниже.

4) Функциональные тестовые файлы (tests/funtional):
4.1) main.cpp - содержит:
- объявление класса TsibarevaERunFuncTestsProcesses;
- переопределение функции PrintTestParam таким образом, что печать тестов содержит размерность матрицы и краткое описание её типа;
- переопределение функции SetUp таким образом, что предположено генерирование матрицы вида, указанного в третьем аргументе тестового типа при использовании метода GenerateMatrixFunc и последовательное вычисление ожидаемых данных при использовании метода GenerateExpectedOutput; необходимо заметить, что класс функциональных тестов TsibarevaERunFuncTestsProcesses не предполагает тестирования на размерах матриц, значительно влияющих на производительность алгоритмов реализации (иными словами, наличие или отсутствие линейного алгоритма генерации вектора ожидаемых значений не влияет существенно на время прохождения тестов); для тестирования производительности алгоритмов была выделена специальная реализация генерации матриц и векторов ожидаемых значений, имеющая меньшую алгоритмическую сложность и идентичная для последовательной и праллельной версии алгоритма;
- инициализацию и заполнение массива аргументов 17 тестовыми случаями, выполнимыми как для последовательной, так и для параллельной реализации алгоритма; тестовые аргументы покрывают такие случаи, как единичная матрица, матрица, состоящая из одной строки/столбца, квадратная матрица, прямоугольная матрица, у которой количество столбцов/строк превышает количество строк/столбцов соответственно; тестовые аргументы не покрывают ситуации, в которых на вход алгоритму передана пустая матрица;
- реализацию 4 дополнительных тестов, проверяющих пустые матрицы, матрицы и состоящие из пустых столбцов для последовательного и параллельного алгоритмов соответственно.

5) Тестовые файлы производительности (tests/performance):
4.1) main.cpp - содержит:
- объявление класса TsibarevaERunPerfTestProcesses;
- переопределение функции SetUp таким образом, что 

## 6. Экспериментальные результаты
Аппаратное обеспечение и характеристики ОС локального запуска:
- Модель процессора: AMD Ryzen 7 5700U (1.80 GHz)
- Архитектура: x86-64
- Ядра: 8 ядер
- Оперативная память: 8 GB
- Операционная система: Windows 10 Home (базовая) / Ubuntu 24.04.3 LTS (сборочная)
- Подсистема: WSL2 (Windows Subsystem for Linux)

Инструменты:
- Компилятор: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- MPI реализация: Open MPI 4.1.6
- Тип сборки: Release 

Настройки окружения:
- PPC_NUM_PROC: 2 (количество MPI процессов)
- PPC_NUM_THREADS: 4 (доступные потоки)

## 7. Результаты и обсуждение

### 7.1 Корректность
Для проверки корректности выполнения алгоритмами вычисления максимальных элементов в столбцах были использованы предварительно подготовленные матрицы и вектора максимумов 17 типов. Каждый тип был прописан в enum MatrixType и предполагает как покрытие различных размерностей матриц, за исключением пустых, так и различных типов их формирования. Так, матрицы могут быть следующих видов:
| Тип                 | Краткое описание                |
|---------------------|---------------------------------|
| kSingleConstant     | 1x1 константная                 |
| kSingleRow          | 1x10 одна строка                |
| kSingleCol          | 3x1 один столбец                |
| kAllZeros           | 5x5 все нули                    |
| kConstant           | 5x5 константная                 |
| kMaxFirst           | 6x4 максимум в первой строке    |
| kMaxLast            | 6x4 максимум в последней строке |
| kMaxMiddle          | 6x4 максимум в середине         |
| kAscending          | 8x8 возрастающая                |
| kDescending         | 8x8 убывающая                   |
| kDiagonalDominant   | 8x8 диагонально доминантная     |
| kSparse             | 8x8 разреженная                 |
| kNegative           | 8x8 отрицательная               |
| kSquareSmall        | 2x2 маленькая квадратная        |
| kVertical           | 10x4 вертикальная               |
| kHorizontal         | 5x10 горизонтальная             |
| kCheckerboard       | 7x7 шахматная                   |

Был предположен отказ от генерации матриц посредством функций в пользу формирования изначально предопределенных наборов данных как для матриц, так и для непосредственно соответствущих этим матрицам векторов максимальных элементов. 
Необходимо отметить, что для формирования наборов тестовых данных тестов производительности ни один из способов формирования изначально предопределенных наборов данных не мог быть корректен в полной мере в силу ограниченности максимальных объемов файлов проекта (затребованны десятки миллионов элементов). Исходя из чего, для performance тестирования была предположена генерация матриц в зависимости от количества строк и столбцов, и генерация ожидаемого вектора максимумов уже в процессе генерации матрицы.

### 7.2 Производительность
Тесты производительности были проведены локально на 36 миллионах элементов.

### Режим Pipeline
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.2983   | 1.00      | N/A           |
| 1 (MPI)   | 0.2982   | 1.00      | 100%          |
| 2 (MPI)   | 0.1907   | 1.56      | 78.2%         |
| 3 (MPI)   | 0.1573   | 1.90      | 63.2%         |
| 4 (MPI)   | 0.1775   | 1.68      | 42.0%         |

### Режим Task Run  
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.2956   | 1.00      | N/A           |
| 1 (MPI)   | 0.2950   | 1.00      | 100%          |
| 2 (MPI)   | 0.1869   | 1.58      | 79.1%         |
| 3 (MPI)   | 0.1365   | 2.17      | 72.2%         |
| 4 (MPI)   | 0.1701   | 1.74      | 43.5%         |

Необходимо отметить, что ускорение каждого запуска было высчитано относительно времени выполнения последовательной версии на одном процессе, а эффективность определена как ускорение, поделенное на количество запущенных процессов.

Можно заметить, что параллельная реализация алгоритмов в обоих режимах показала ускорение от 1.56 до 2.17 раз относительно последовательной. Несмотря на это, локальным тестированием на относительно небольшом количестве элементов (десятки, но не сотни миллионов) были обусловлены издержки эффективности на накладные расходы параллельного алгоритма с использованием OpenMPI, которые показали линейный рост с увеличением числа процессов. 

## 8. Выводы
Задача реализации алгоритма поиска максимумов в матрице по столбцам была решена последовательным образом и параллельным образом при использовании OpenMPI. Были предложены различные способы работы формирования исходных данных для функциональных тестов и тестов производительности. А также выполнены замеры времени выполнения последовательного и параллельного алгоритмов с последующим высчитыванием эффективности распараллеливания вычислений.
Замеры выполнения алгоритмов показали ускорение выполнения задачи в ~1,5-2,7 раз при использовании параллельных вычислений. Циклическая схема распределения данных при этом предполагает балансировку нагрузки. 

## 9. Источники

1. **Технологии параллельного программирования MPI и OpenMP** // А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.

2. **Инструменты параллельного программирования в системах с общей памятью: Учебное пособие.** // Корняков К.В., Мееров И.Б., Сиднев А.А., Сысоев А.В., Шишков А.В., - Нижний Новгород: Изд-во Нижегородского госуниверситета, 2010. - 202 с.

3. **Справочник по MPI** // URL: https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference, 2023 (дата обращения: 19.11.2025).

4. **Open MPI: Open Source High Performance Computing** // URL: https://www-lb.open-mpi.org/doc/v4.1, 2025 (дата обращения: 19.11.2025).

### 10. Приложения

## Приложение 1: код реализации последовательного алгоритма
``` cpp
bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  auto &column_maxs = GetOutput();
  size_t cols_count = matrix[0].size();

  for (size_t col = 0; col < cols_count; ++col) {
    int max_value = matrix[0][col];
    for (size_t row = 1; row < matrix.size(); ++row) {
      max_value = std::max(matrix[row][col], max_value);
    }
    column_maxs[col] = max_value;
  }

  return true;
}

## Приложение 2: код реализации параллельного алгоритма
``` cpp
bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &matrix = GetInput();
  size_t rows_count = matrix.size();
  size_t cols_count = matrix[0].size();

  std::vector<int> local_maxs;

  for (auto col = static_cast<size_t>(world_rank); col < cols_count; col += static_cast<size_t>(world_size)) {
    int max_val = matrix[0][col];
    for (size_t row = 1; row < rows_count; ++row) {
      max_val = std::max(matrix[row][col], max_val);
    }
    local_maxs.push_back(max_val);
  }

  if (world_rank == 0) {
    CollectResultsFromAllProcesses(local_maxs, world_size, cols_count);
  } else {
    if (!local_maxs.empty()) {
      MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

void TsibarevaEMatrixColumnMaxMPI::CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size,
                                                                  size_t cols_count) {
  final_result_.resize(cols_count);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t idx = 0;
  for (size_t col = 0; col < cols_count && idx < local_maxs.size(); col += world_size) {
    final_result_[col] = local_maxs[idx++];
  }

  for (int proc = 1; proc < world_size; proc++) {
    int proc_pass = 0;

    for (size_t col = proc; col < cols_count; col += world_size) {
      proc_pass++;
    }

    if (proc_pass <= 0) {
      continue;
    }

    std::vector<int> proc_maxs(static_cast<size_t>(proc_pass));
    MPI_Recv(proc_maxs.data(), proc_pass, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size_t proc_idx = 0;
    for (size_t col = proc; col < cols_count; col += world_size) {
      final_result_[col] = proc_maxs[proc_idx++];
    }
  }
}

## Приложение 3: генерация данных для функциональных тестов
``` cpp
inline std::vector<std::vector<int>> GenerateSingleConstantMatrix() {
  return {{30}};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {30};
}

inline std::vector<std::vector<int>> GenerateSingleRowMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
}

inline std::vector<std::vector<int>> GenerateSingleColMatrix() {
  return {{1}, {2}, {3}};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3};
}

inline std::vector<std::vector<int>> GenerateAllZerosMatrix() {
  return {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
  };
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0, 0};
}

inline std::vector<std::vector<int>> GenerateConstantMatrix() {
  return {
    {30, 30, 30, 30, 30},
    {30, 30, 30, 30, 30},
    {30, 30, 30, 30, 30},
    {30, 30, 30, 30, 30},
    {30, 30, 30, 30, 30}
  };
}
inline std::vector<int> GenerateConstantExpected() {
  return {30, 30, 30, 30, 30};
}

inline std::vector<std::vector<int>> GenerateMaxFirstMatrix() {
  return {
    {1000, 1001, 1002, 1003},  
    {1, 2, 3, 4},
    {2, 3, 4, 5},
    {3, 4, 5, 6},
    {4, 5, 6, 7},
    {5, 6, 7, 8}
  };
}
inline std::vector<int> GenerateMaxFirstExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxLastMatrix() {
  return {
    {1, 2, 3, 4},
    {2, 3, 4, 5},
    {3, 4, 5, 6},
    {4, 5, 6, 7},
    {5, 6, 7, 8},
    {1000, 1001, 1002, 1003}  
  };
}
inline std::vector<int> GenerateMaxLastExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxMiddleMatrix() {
  return {
    {1, 2, 3, 4},
    {2, 3, 4, 5},
    {1000, 1001, 1002, 1003}, 
    {4, 5, 6, 7},
    {5, 6, 7, 8},
    {6, 7, 8, 9}
  };
}
inline std::vector<int> GenerateMaxMiddleExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateAscendingMatrix() {
  return {
    {1, 2, 3, 4, 5, 6, 7, 8},
    {9, 10, 11, 12, 13, 14, 15, 16},
    {17, 18, 19, 20, 21, 22, 23, 24},
    {25, 26, 27, 28, 29, 30, 31, 32},
    {33, 34, 35, 36, 37, 38, 39, 40},
    {41, 42, 43, 44, 45, 46, 47, 48},
    {49, 50, 51, 52, 53, 54, 55, 56},
    {57, 58, 59, 60, 61, 62, 63, 64}
  };
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}

inline std::vector<std::vector<int>> GenerateDescendingMatrix() {
  return {
    {64, 63, 62, 61, 60, 59, 58, 57},
    {56, 55, 54, 53, 52, 51, 50, 49},
    {48, 47, 46, 45, 44, 43, 42, 41},
    {40, 39, 38, 37, 36, 35, 34, 33},
    {32, 31, 30, 29, 28, 27, 26, 25},
    {24, 23, 22, 21, 20, 19, 18, 17},
    {16, 15, 14, 13, 12, 11, 10, 9},
    {8, 7, 6, 5, 4, 3, 2, 1}
  };
}
inline std::vector<int> GenerateDescendingExpected() {
  return {64, 63, 62, 61, 60, 59, 58, 57};
}

inline std::vector<std::vector<int>> GenerateDiagonalDominantMatrix() {
  return {
    {1000, 1, 2, 3, 4, 5, 6, 7},
    {1, 1100, 3, 4, 5, 6, 7, 8},
    {2, 3, 1200, 5, 6, 7, 8, 9},
    {3, 4, 5, 1300, 7, 8, 9, 10},
    {4, 5, 6, 7, 1400, 9, 10, 11},
    {5, 6, 7, 8, 9, 1500, 11, 12},
    {6, 7, 8, 9, 10, 11, 1600, 14},
    {7, 8, 9, 10, 11, 12, 14, 1700}
  };
}
inline std::vector<int> GenerateDiagonalDominantExpected() {
  return {1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700};
}

inline std::vector<std::vector<int>> GenerateSparseMatrix() {
  return {
    {8, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 61, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 72, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 9, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}
  };
}
inline std::vector<int> GenerateSparseExpected() {
  return {8, 9, 0, 61, 0, 0, 72, 0};
}

inline std::vector<std::vector<int>> GenerateNegativeMatrix() {
  return {
    {-30, -31, -32, -33, -34, -35, -36, -37},
    {-40, -41, -42, -43, -44, -45, -46, -47},
    {-50, -51, -52, -53, -54, -55, -56, -57},
    {-60, -61, -62, -63, -64, -65, -66, -67},
    {-70, -71, -72, -73, -74, -75, -76, -77},
    {-80, -81, -82, -83, -84, -85, -86, -87},
    {-90, -91, -92, -93, -94, -95, -96, -97},
    {-100, -101, -102, -103, -104, -105, -106, -107}
  };
}
inline std::vector<int> GenerateNegativeExpected() {
  return {-30, -31, -32, -33, -34, -35, -36, -37};
}

inline std::vector<std::vector<int>> GenerateSquareSmallMatrix() {
  return {
    {1, 2},
    {3, 4}
  };
}
inline std::vector<int> GenerateSquareSmallExpected() {
  return {3, 4};
}

inline std::vector<std::vector<int>> GenerateVerticalMatrix() {
  return {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16},
    {17, 18, 19, 20},
    {21, 22, 23, 24},
    {25, 26, 27, 28},
    {29, 30, 31, 32},
    {33, 34, 35, 36},
    {37, 38, 39, 40}
  };
}
inline std::vector<int> GenerateVerticalExpected() {
  return {37, 38, 39, 40};
}

inline std::vector<std::vector<int>> GenerateHorizontalMatrix() {
  return {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
    {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
    {41, 42, 43, 44, 45, 46, 47, 48, 49, 50}
  };
}
inline std::vector<int> GenerateHorizontalExpected() {
  return {41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
}

inline std::vector<std::vector<int>> GenerateCheckerboardMatrix() {
  return {
    {1, -1, 1, -1, 1, -1, 1},
    {-1, 1, -1, 1, -1, 1, -1},
    {1, -1, 1, -1, 1, -1, 1},
    {-1, 1, -1, 1, -1, 1, -1},
    {1, -1, 1, -1, 1, -1, 1},
    {-1, 1, -1, 1, -1, 1, -1},
    {1, -1, 1, -1, 1, -1, 1}
  };
}
inline std::vector<int> GenerateCheckerboardExpected() {
  return {1, 1, 1, 1, 1, 1, 1};
}

inline std::vector<std::vector<int>> GenerateMatrixFunc(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:    return GenerateSingleConstantMatrix();
    case MatrixType::kSingleRow:         return GenerateSingleRowMatrix();
    case MatrixType::kSingleCol:         return GenerateSingleColMatrix();
    case MatrixType::kAllZeros:          return GenerateAllZerosMatrix();
    case MatrixType::kConstant:          return GenerateConstantMatrix();
    case MatrixType::kMaxFirst:          return GenerateMaxFirstMatrix();
    case MatrixType::kMaxLast:           return GenerateMaxLastMatrix();
    case MatrixType::kMaxMiddle:         return GenerateMaxMiddleMatrix();
    case MatrixType::kAscending:         return GenerateAscendingMatrix();
    case MatrixType::kDescending:        return GenerateDescendingMatrix();
    case MatrixType::kDiagonalDominant:  return GenerateDiagonalDominantMatrix();
    case MatrixType::kSparse:            return GenerateSparseMatrix();
    case MatrixType::kNegative:          return GenerateNegativeMatrix();
    case MatrixType::kSquareSmall:       return GenerateSquareSmallMatrix();
    case MatrixType::kVertical:          return GenerateVerticalMatrix();
    case MatrixType::kHorizontal:        return GenerateHorizontalMatrix();
    case MatrixType::kCheckerboard:      return GenerateCheckerboardMatrix();
  }
  return GenerateSingleConstantMatrix(); 
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:    return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:         return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:         return GenerateSingleColExpected();
    case MatrixType::kAllZeros:          return GenerateAllZerosExpected();
    case MatrixType::kConstant:          return GenerateConstantExpected();
    case MatrixType::kMaxFirst:          return GenerateMaxFirstExpected();
    case MatrixType::kMaxLast:           return GenerateMaxLastExpected();
    case MatrixType::kMaxMiddle:         return GenerateMaxMiddleExpected();
    case MatrixType::kAscending:         return GenerateAscendingExpected();
    case MatrixType::kDescending:        return GenerateDescendingExpected();
    case MatrixType::kDiagonalDominant:  return GenerateDiagonalDominantExpected();
    case MatrixType::kSparse:            return GenerateSparseExpected();
    case MatrixType::kNegative:          return GenerateNegativeExpected();
    case MatrixType::kSquareSmall:       return GenerateSquareSmallExpected();
    case MatrixType::kVertical:          return GenerateVerticalExpected();
    case MatrixType::kHorizontal:        return GenerateHorizontalExpected();
    case MatrixType::kCheckerboard:      return GenerateCheckerboardExpected();
  }
  return GenerateSingleConstantExpected(); 
}
