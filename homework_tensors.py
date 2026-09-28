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

# Проверка размеров и значений
assert tensor_random.shape == (3, 4)
assert torch.all((tensor_random >= 0) & (tensor_random <= 1))
assert tensor_zeros.shape == (2, 3, 4)
assert torch.all(tensor_zeros == 0)
assert tensor_ones.shape == (5, 5)
assert torch.all(tensor_ones == 1)
assert tensor_0_15.shape == (4, 4)
assert torch.equal(tensor_0_15.flatten(), torch.arange(16))

print("3x4 random tensor:\n", tensor_random)
print("2x3x4 zeros tensor:\n", tensor_zeros)
print("5x5 ones tensor:\n", tensor_ones)
print("4x4 tensor from 0 to 15:\n", tensor_0_15)


# 1.2 Операции с тензорами

A = torch.rand(3, 4)
B = torch.rand(4, 3)

# Проверка типов и размерностей
if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
    raise TypeError("A и B должны быть тензорами PyTorch.")

if A.ndim != 2 or B.ndim != 2:
    raise ValueError("A и B должны быть двумерными тензорами.")

if A.shape[1] != B.shape[0]:
    raise ValueError("Размерности A и B несовместимы для матричного умножения.")

if A.shape != B.T.shape:
    raise ValueError(
        "Размерности A и B.T несовместимы для поэлементного умножения."
    )

# Транспонирование A
A_transposed = A.T

# Матричное умножение A и B
matrix_product = torch.matmul(A, B)

# Поэлементное умножение A и транспонированного B
elementwise_product = A * B.T

# Сумма всех элементов A
sum_A = torch.sum(A)

# Простые тесты результатов
assert A_transposed.shape == (4, 3)
assert matrix_product.shape == (3, 3)
assert elementwise_product.shape == (3, 4)
assert sum_A.ndim == 0

print("A:\n", A)
print("B:\n", B)
print("A transposed:\n", A_transposed)
print("A x B:\n", matrix_product)
print("A * B.T:\n", elementwise_product)
print("Sum of A:", sum_A)


# 1.3 Индексация и срезы

tensor_3d = torch.arange(125).reshape(5, 5, 5)

if tensor_3d.shape != (5, 5, 5):
    raise ValueError("Ожидался тензор размером 5x5x5.")

# Первая строка каждого слоя
first_row = tensor_3d[:, 0, :]

# Последний столбец каждого слоя
last_column = tensor_3d[:, :, -1]

# Центральная область 2x2 каждого слоя.
# Для матрицы 5x5 центральная область четного размера неоднозначна,
# поэтому выбираем строки и столбцы с индексами 1 и 2.
center_submatrix = tensor_3d[:, 1:3, 1:3]

# Элементы с четными индексами по каждой размерности
even_indices = tensor_3d[::2, ::2, ::2]

# Проверка размеров срезов
assert first_row.shape == (5, 5)
assert last_column.shape == (5, 5)
assert center_submatrix.shape == (5, 2, 2)
assert even_indices.shape == (3, 3, 3)

print("First row:\n", first_row)
print("Last column:\n", last_column)
print("Center 2x2 submatrix:\n", center_submatrix)
print("Elements with even indices:\n", even_indices)


# 1.4 Работа с формами

tensor_24 = torch.arange(24)

if tensor_24.numel() != 24:
    raise ValueError("Тензор должен содержать ровно 24 элемента.")

shape_2x12 = tensor_24.reshape(2, 12)
shape_3x8 = tensor_24.reshape(3, 8)
shape_4x6 = tensor_24.reshape(4, 6)
shape_2x3x4 = tensor_24.reshape(2, 3, 4)
shape_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)

# Проверка новых форм
assert shape_2x12.shape == (2, 12)
assert shape_3x8.shape == (3, 8)
assert shape_4x6.shape == (4, 6)
assert shape_2x3x4.shape == (2, 3, 4)
assert shape_2x2x2x3.shape == (2, 2, 2, 3)

print("2x12:\n", shape_2x12)
print("3x8:\n", shape_3x8)
print("4x6:\n", shape_4x6)
print("2x3x4:\n", shape_2x3x4)
print("2x2x2x3:\n", shape_2x2x2x3)

print("\nВсе проверки homework_tensors.py успешно пройдены.")