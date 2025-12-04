# Нахождение максимальных значений по столбцам матрицы

- Студент: Цибарева Екатерина Алексеевна, группа 3823Б1ПР1
- Технология: SEQ | MPI
- Вариант: 16

## 1. Введение
Задача поиска максимумов по столбцам матрицы критически важна для анализа данных в машинном обучении, финансовом моделировании и научных вычислениях. При работе с матрицами большого размера последовательные алгоритмы показывают не удовлетворительные результаты, критически сказывающиеся на общей производительности вычислений.

Целью данной работы является реализация и сравнительный анализ последовательного и параллельного алгоритмов для нахождения максимальных значений и их позиций в каждом столбце матрицы произвольной размерности.

## 2. Постановка задачи
**Описание задачи**
Для каждого столбца заданной матрицы найти значение максимального элемента.\

**Входной тип данных** 
- Вектор значений типа int (представляющий матрицу, хранимую по столбцам (column-major)).
- Количество строк.
- Количество столбцов.

Удобство хранения именно в column-major порядке обусловлено необходимостью рассылки плоского массива в виде непрерывных интервалов памяти. Задача перевода из row-major в column-major порядок хранения матрицы может быть решена отдельно, вне последовательной и параллельной реализации решения задачи.\

``` cpp
using InType = std::tuple<std::vector<int>, int, int>;
```

**Выходной тип данных** 
- Вектор значений типа int, содержащий максимумы по столбцам.

``` cpp
using OutType = std::vector<int>;
```

**Ограничения**
- матрица может состоять из повторяющихся значений либо содержать повторяющиеся значения;
- результирующие вектора последовательной и параллельной реализаций алгоритма не должны различаться;
- для реализации параллельного алгоритма должен быть использован MPI;
- матрица может быть квадратной или прямоугольной;
- матрица может быть подана на вход алгоритма в не корректном формате, а именно быть пустой или содержать только пустые столбцы.

## 3. Описание базового алгоритма
**Основная идея**

Базовый, или последовательный, алгоритм выполняет поиск максимальных значений для каждого столбца матрицы путем последовательного обхода всех элементов. Необходимо для каждого столбца независимо найти наибольший элемент, просматривая все элементы этого столбца.

Конвейер выполнения как последовательного, так и параллельного алгоритма предполагает обязательную проверку на не пустоту матрицы и не пустоту её столбцов. 
``` cpp
if (rows_ == 0 || cols_ == 0) {
    return true;
}
```

**Шаги базового (последовательного) алгоритма обработки**: 

1. Инициализация матрицы
``` cpp
const auto &matrix = GetInput();
```
2. Обязательная проверка на пустоту матриц, штатная ситуация возращаемого значения для пустых матриц - возвращение пустого вектора.
3. Инициализация выходного вектора
``` cpp
auto &column_maxs = GetOutput();
```
4. Алгоритм предполагает последовательный обход всех элементов с квадратичной (m*n, где m и n - количество строк и столбцов соответственно) сложностью. В качестве первоначально присваиваемого максимального значения выбран элемент первой строки в цикле.

Частичный код базового (последовательного) алгоритма поиска максимумов матрицы по столбцам можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

  auto &column_maxs = GetOutput();

  for (int col = 0; col < cols_; ++col) {
    int maxum_value = input_matrix_[static_cast<size_t>(col) * rows_];
    for (int row = 1; row < rows_; ++row) {
      int idx = (col * rows_) + row;
      int element = input_matrix_[idx];
      maxum_value = std::max(element, maxum_value);
    }
    column_maxs[col] = maxum_value;
  }

  return true;
}
```

## 4. Схема распараллеливания
**Модель распределения**

Предположено получение каждым процессом равного числа столбцов, первые процессы при этом делят между собой остаток, получая на 1 столбец больше остальных. Столбцы распределяются последовательно, с возрастанием смещения в массиве, и, соответственно, номера столбца, в порядке возрастания ранга процесса (процесс с большим рангом получает столбцы с большим номером последовательным образом).

**Роли процессов**

Процесс 0 выступает в роли координатора:
- распределяет и рассылает данные матрицы в количестве определенного числа столбцов на процесс;
- собирает частичные результаты с процессов рангом 1 и выше;
- формирует результирующий вектор в соответствии с моделью рапределения столбцов между процессами;
- рассылает финальный результат.

Процессы рангом 1 и выше:
- выполняют локальные вычисления максимумов для назначенных и направленных им столбцов;
- передают результаты локальных вычислений процессу с рангом 0.

**Коммуникационная схема**

**Шаги параллельного алгоритма обработки**: 

0. Подготовительные действия на процессе 0: расчет смещений в исходном плоском массиве, представляющем матрицу, и рассылка массивов, содержащих смещения и количества элементов на процесс, - всем запущенным процессам. 

1. Рассылка данных на каждый процесс посредством MPI_Scatterv:
``` cpp
void TsibarevaEMatrixColumnMaxMPI::ScatterMatrixData(int world_rank, const std::vector<int> &send_counts,
                                                 const std::vector<int> &displacements) {
  local_flat_data_.resize(static_cast<size_t>(local_cols_) * rows_);
  MPI_Scatterv(world_rank == 0 ? input_matrix_.data() : nullptr,
             send_counts.data(),
             displacements.data(),
             MPI_INT,
             local_flat_data_.data(),
             static_cast<int>(local_flat_data_.size()),
             MPI_INT,
             0,
             MPI_COMM_WORLD);
}
```
3. Подсчет локальных максимумов по выделенным процессу столбцам:
``` cpp
std::vector<int> TsibarevaEMatrixColumnMaxMPI::CalculateLocalColumnMaxima() {
  std::vector<int> local_maxs(local_cols_, std::numeric_limits<int>::min());

  for (int col = 0; col < local_cols_; col++) {
    for (int row = 0; row < rows_; row++) {
      int idx = (col * rows_) + row;
      local_maxs[col] = std::max(local_flat_data_[idx], local_maxs[col]);
    }
  }

  return local_maxs;
}
```
5. Сбор результата на процессе 0 и синхронизация посредством MPI_Allgatherv:
``` cpp
MPI_Allgatherv(local_maxs.data(),
               local_cols_,
               MPI_INT,
               global_result.data(),
               recv_counts.data(),
               displs.data(),
               MPI_INT,
               MPI_COMM_WORLD);
```

6. Установление GetOutput корректного значения результата.

**Особенности модели распределения и коммуникационной схемы**:
- распределение данных между процессами последовательно по столбцам;
- выполнение распределения, рассылок и формирования результата первично - на нулевом (координаторском) процессе;
- эффективный сбор и синхронизация данных на всех процессах одновременно.

Частичный код реализации алгоритма параллельной обработки можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastMatrixDimensions();

  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  int cols_base = cols_ / world_size;
  int remainder = cols_ % world_size;
  local_cols_ = cols_base + (world_rank < remainder ? 1 : 0);

  std::vector<int> send_counts;
  std::vector<int> displacements;
  PrepareScatterParameters(world_rank, world_size, send_counts, displacements);

  ScatterMatrixData(world_rank, send_counts, displacements);

  std::vector<int> local_maxs = CalculateLocalColumnMaxima();

  int cols_base = cols_ / world_size;
  int remainder = cols_ % world_size;

  std::vector<int> recv_counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<int> global_result(cols_);

  int mdisplace = 0;
  for (int i = 0; i < world_size; i++) {
    int proc_cols = cols_base + (i < remainder ? 1 : 0);
    recv_counts[i] = proc_cols;
    displs[i] = mdisplace;
    mdisplace += proc_cols;
  }

  MPI_Allgatherv(local_maxs.data(), local_cols_, MPI_INT, global_result.data(), recv_counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}
```

## 5. Детали реализации

**Структура проекта**
``` text
tsibareva_e_matrix_column_max
        │   info.json
        │   report.md
        │   settings.json
        ├───common
        │   └───include
        │           common.hpp
        ├───data
        │       pic.jpg
        ├───mpi
        │   ├───include
        │   │       ops_mpi.hpp
        │   │
        │   └───src
        │           ops_mpi.cpp
        ├───seq
        │   ├───include
        │   │       ops_seq.hpp
        │   │
        │   └───src
        │           ops_seq.cpp
        └───tests
            │   .clang-tidy
            │
            ├───functional
            │       main.cpp
            └───performance
                    main.cpp
```

Проект реализации исходных алгоритмов, функционального тестирования и тестирования производительности имеет сложную структуру. 


**Файлы реализаций**

1. Последовательная реализация (seq):

- ops_seq.hpp - объявление класса TsibarevaEMatrixColumnMaxSEQ, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl());

- ops_seq.cpp - реализация перегруженных методов:
    - RunImpl()            - базовый (последовательный) алгоритм поиска максимумов;
    - PreProcessingImpl()  - инициализирован выходной вектор, проверены случаи не корректного задания матрицы (пустота);
    - ValidationImpl()     - не предположено отдельной логики;
    - PostProcessingImpl() - не предположено отдельной логики.

2. MPI реализация (mpi):

- ops_mpi.hpp - объявление класса TsibarevaEMatrixColumnMaxMPI, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl()), объявлен вектор final_result_;

- ops_mpi.cpp - реализация методов:
    - ValidationImpl()              - не предположено отдельной логики;
    - PreProcessingImpl()           - не предположено отдельной логики;
    - RunImpl()                     - инициализирован выходной вектор, проверены случаи не корректного задания матрицы (пустота), реализован параллельный алгоритм поиска максимумов, распределяющий содержательные действия между процессами;
    - PostProcessingImpl()          - не предположено отдельной логики;
    - PrepareScatterParameters(...) - подготовлены и разосланы вектора смещений в исходном плоском векторе для последующей рассылки данных с нулевого процесса по остальным запущенным;
    - ScatterMatrixData(...)        - разосланы данные исходной матрицы в количестве нескольких столбцов на запущенный процесс;
    - CalculateLocalColumnMaxima()  - рассчитаны локальные значения максимумов по полученным столбцам.

Для удобства работы были введены глобальные переменные как для хранения количества строк, столбцов и количества обрабатываемых локально столбцов, так и для хранения локальных столбцов в векторе local_flat_data_.

**Тестирование и формирование входных данных**

3. Общие компоненты (common):
- common.hpp содержит:
    - объявление входного (InType), выходного (OutType) типа, типа базовой задачи (BaseTask) и тестовых классов (TestTask); необходимо отметить, что входные данные представлены в формате вектора векторов, выходные - вектора максимумов, а тестовый класс состоит из содержательного типа матрицы и строки её краткого описания; 
    ``` cpp
    using InType = std::tuple<std::vector<int>, int, int>;
    using OutType = std::vector<int>;
    using TestType = std::tuple<MatrixType, std::string>;
    using BaseTask = ppc::task::Task<InType, OutType>;
    ``` 
    - перечисление типов матриц, доступных для генерации (enum MatrixType), более подробно описанных в пункте 7.1 "Корректность";
    - функции формирования матриц 19 типов в соответствии с перечислением;
    - функцию задания матрицы GenerateMatrixFunc, в качестве аргументов принимающую тип генерируемой матрицы, и принимающей решение о вызове соответствующей функции генерации матрицы;
    - функцию GenerateExpectedOutput, задающую эталонный вектор выходных значений максимумов по столбцам для каждого типа матрицы.

Необходимо отметить, что функции формирования матриц были реализованы таким образом, что задают заранее известные наборы данных без генерации произвольных матриц. Выбор был сделан в пользу предопределенных сформированных матриц и векторов с целью наиболее точного сравнения результатов с известным эталонным значением. Например, генерация возрастающей квадратной матрицы 8 на 8 и её эталонного вектора представлена ниже.
``` cpp
inline std::tuple<std::vector<int>, int, int> GenerateAscendingMatrix() {
  std::vector<int> matrix = {1,  9,  17, 25, 33, 41, 49, 57,
                             2,  10, 18, 26, 34, 42, 50, 58,
                             3,  11, 19, 27, 35, 43, 51, 59,
                             4,  12, 20, 28, 36, 44, 52, 60,
                             5,  13, 21, 29, 37, 45, 53, 61,
                             6,  14, 22, 30, 38, 46, 54, 62,
                             7,  15, 23, 31, 39, 47, 55, 63,
                             8,  16, 24, 32, 40, 48, 56, 64};
  return {matrix, 8, 8};
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}
```

4. Функциональные тестовые файлы (tests/funtional):
- main.cpp - содержит:
    - объявление класса TsibarevaERunFuncTestsProcesses;
    - переопределение функции PrintTestParam таким образом, что печать тестов содержит краткое описание её типа, включающее размерность;
    - переопределение функции SetUp таким образом, что исходная матрица может быть получена как GenerateMatrixFunc(<тип матрицы 1 из 20 перечисленных>), а ожидаемые данные - как GenerateExpectedOutput(<тот же тип матрицы>);
    ``` cpp
    MatrixType matrix_type = std::get<0>(params);
    input_data_            = GenerateMatrixFunc(matrix_type);
    expected_output_       = GenerateExpectedOutput(matrix_type);
    ```
    - инициализацию и заполнение массива аргументов 19 тестовыми случаями, выполнимыми как для последовательной, так и для параллельной реализации алгоритма; тестовые аргументы покрывают такие случаи, как единичная матрица, матрица, состоящая из одной строки/столбца, квадратная матрица, прямоугольная матрица, у которой количество столбцов/строк превышает количество строк/столбцов соответственно; тестовые аргументы покрывают ситуации, в которых на вход алгоритму передана пустая матрица.

5. Тестовые файлы производительности (tests/performance):
- main.cpp - содержит:
    - объявление класса TsibarevaERunPerfTestProcesses;
    - переопределение функции SetUp таким образом, что исходная матрица и эталонный вектор (опциональный в случае нагрузочного тестирования) максимумов по столбцам в ней формируются на фиксированное количество элементов, размером 20000 на 20000 элементов для запуска Github Actions и 6000 на 6000 - для локального запуска. 

Необходимо отметить, что в случае генерации объемных матриц для тестирования производительности сформирована квадратная матрица с максимумом в середине столбца. Кроме того, ни одна реализация формирования тестовых данных не использует случайную генерацию чисел ни в каком виде, что достаточно для покрытия основных сценариев работы и предполагает стабильное выполнение алгоритма при многократных запусках.

## 6. Экспериментальные результаты

**Аппаратное обеспечение и характеристики ОС локального запуска**:
- Модель процессора: AMD Ryzen 7 5700U (1.80 GHz)
- Архитектура: x86-64
- Ядра: 8 ядер
- Оперативная память: 8 GB
- Операционная система: Windows 10 Home (базовая) / Ubuntu 24.04.3 LTS (сборочная)
- Подсистема: WSL2 (Windows Subsystem for Linux)

**Инструменты**:
- Компилятор: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- MPI реализация: Open MPI 4.1.6
- Тип сборки: Release 

**Настройки окружения**:
- PPC_NUM_PROC: 2 (количество MPI процессов)
- PPC_NUM_THREADS: 4 (доступные потоки)
- Количество доступных процессов PPC_NUM_PROC также может быть задано ключом ``` mpirun -n M ``` при запуске тестов, где M - требуемое количество процессов.

## 7. Результаты и обсуждение

### 7.1 Корректность
Для проверки корректности выполнения алгоритмами вычисления максимальных элементов в столбцах были использованы предварительно подготовленные матрицы и эталонные вектора максимумов 20 типов. Каждый тип был прописан в enum MatrixType и предполагает как покрытие различных размерностей матриц, так и различных типов их формирования. Так, тестовые матрицы могут быть следующих видов:
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
| kEmpty              | Пустая матрица                  |
| kZeroColumns        | Матрица с пустыми столбцами     |

Был предположен отказ от генерации матриц посредством функций в пользу формирования изначально предопределенных наборов данных для матриц и эталонных векторов. Эталонные вектора, возвращаемые по аргументу типа соответствующего типу формирования матрицы, содержат максимумы этих матриц. 

Несмотря на то, что для функционального тестирования метод задания предопределённых данных корректен, в силу ограниченности максимальных объемов файлов проекта такой способ не может быть применим к генерации данных для тестирования производительности. Исходя из чего, для performance тестирования была предположена генерация матриц в зависимости от количества строк и столбцов (по 6 или 10 тысяч), и подсчет эталонного вектора максимумов в процессе генерации матрицы.

Полный код формирования тестовых случаев функционального тестирования можно видеть в приложении 1.

### 7.2 Производительность
Тесты производительности были проведены локально на 36 миллионах элементов.

### Режим Pipeline
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.062    | 1.00      | N/A           |
| 1 (MPI)   | 0.058    | 1.069     | 106.9%        |
| 2 (MPI)   | 0.055    | 1.127     | 56.4%         |
| 3 (MPI)   | 0.051    | 1.216     | 40.5%         |
| 4 (MPI)   | 0.052    | 1.192     | 29.8%         |

### Режим Task Run  
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.051    | 1.00      | N/A           |
| 1 (MPI)   | 0.059    | 0.864     | 86.4%         |
| 2 (MPI)   | 0.057    | 0.895     | 44.7%         |
| 3 (MPI)   | 0.053    | 0.962     | 32.1%         |
| 4 (MPI)   | 0.052    | 0.981     | 24.5%         |

Таким образом, параллельная реализация показала максимальное ускорение в 1.216 раз на 3 процессах в pipeline режиме, а также некоторое ухудшение показателя эффективности (ускорение к количеству процессов) и незначительное увеличение ускорения с ростом количества процессов. 

И то и другое обусловлено ростом накладных расходов пропорционально вырабатываемому ускорению реализации (вследствие чего время выполнения уменьшается, но не значительно, ускорение остается близко к 1.0), который не может быть покрыт количеством элементов матрицы 6000^2 элементов, из-за чего эффективность выполнения показывает линейное ухудшение с линейным ростом количества процессов. 

Основным фактором значительного роста накладных расходов можно считать использование MPI_Scatterv для организации обработки данных на распределённой памяти. 

## 8. Выводы
Задача реализации алгоритма поиска максимумов в матрице по столбцам была решена последовательным образом и параллельным образом при использовании OpenMPI. Были предложены различные способы формирования исходных данных для функциональных тестов и тестов производительности. А также выполнены замеры времени выполнения последовательного и параллельного алгоритмов с последующим высчитыванием эффективности распараллеливания вычислений.

Выявлено линейное ухудшение эффективности в силу скорости роста накладных расходов при использовании MPI_Scatterv для рассылки данных.
Достигнуто ускорение параллельной реализации 1.216 при обработке на 3 процессах.

## 9. Источники

1. **Технологии параллельного программирования MPI и OpenMP** // А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.
2. **Инструменты параллельного программирования в системах с общей памятью: Учебное пособие.** // Корняков К.В., Мееров И.Б., Сиднев А.А., Сысоев А.В., Шишков А.В., - Нижний Новгород: Изд-во Нижегородского госуниверситета, 2010. - 202 с.
3. **Справочник по MPI** // URL: https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference, 2023 (дата обращения: 15.11.2025).
4. **Open MPI: Open Source High Performance Computing** // URL: https://www-lb.open-mpi.org/doc/v4.1, 2025 (дата обращения: 15.11.2025).

### 10. Приложения

## Приложение 1: генерация данных для функциональных тестов
``` cpp
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
```