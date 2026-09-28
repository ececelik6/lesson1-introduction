import torch

# 1.1 Создание тензоров

# Тензор 3x4 со случайными числами от 0 до 1
tensor_random = torch.rand(3, 4)

# Тензор 2x3x4, заполненный нулями
tensor_zeros = torch.zeros(2, 3, 4)

# Тензор 5x5, заполненный единицами
tensor_ones = torch.ones(5, 5)

# Тензор 4x4 с числами от 0 до 15
tensor_0_15 = torch.arange(16).reshape(4, 4)

print("3x4 random tensor:\n", tensor_random)
print("2x3x4 zeros tensor:\n", tensor_zeros)
print("5x5 ones tensor:\n", tensor_ones)
print("4x4 tensor from 0 to 15:\n", tensor_0_15)


# 1.2 Операции с тензорами

# Тензор A размером 3x4
A = torch.rand(3, 4)

# Тензор B размером 4x3
B = torch.rand(4, 3)

# Транспонирование A
A_transposed = A.T

# Матричное умножение A и B
matrix_product = torch.matmul(A, B)

# Поэлементное умножение A и транспонированного B
elementwise_product = A * B.T

# Сумма всех элементов A
sum_A = torch.sum(A)

print("A:\n", A)
print("B:\n", B)
print("A transposed:\n", A_transposed)
print("A x B:\n", matrix_product)
print("A * B.T:\n", elementwise_product)
print("Sum of A:", sum_A)


# 1.3 Индексация и срезы

# Тензор размером 5x5x5
tensor_3d = torch.arange(125).reshape(5, 5, 5)

# Первая строка
first_row = tensor_3d[:, 0, :]

# Последний столбец
last_column = tensor_3d[:, :, -1]

# Подматрица 2x2 из центра
center_submatrix = tensor_3d[:, 1:3, 1:3]

# Элементы с четными индексами
even_indices = tensor_3d[::2, ::2, ::2]

print("First row:\n", first_row)
print("Last column:\n", last_column)
print("Center 2x2 submatrix:\n", center_submatrix)
print("Elements with even indices:\n", even_indices)


# 1.4 Работа с формами

# Тензор из 24 элементов
tensor_24 = torch.arange(24)

# Изменение формы тензора
shape_2x12 = tensor_24.reshape(2, 12)
shape_3x8 = tensor_24.reshape(3, 8)
shape_4x6 = tensor_24.reshape(4, 6)
shape_2x3x4 = tensor_24.reshape(2, 3, 4)
shape_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)

print("2x12:\n", shape_2x12)
print("3x8:\n", shape_3x8)
print("4x6:\n", shape_4x6)
print("2x3x4:\n", shape_2x3x4)
print("2x2x2x3:\n", shape_2x2x2x3)