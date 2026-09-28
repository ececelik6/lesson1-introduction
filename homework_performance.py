import torch
import time

# 3.1 Подготовка данных

# Проверяем доступность CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна. Тесты будут выполнены только на CPU.")


# Размеры матриц
matrix_sizes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256)
]

# Создаем матрицы на CPU
matrices = []

for size in matrix_sizes:
    matrix = torch.rand(size)
    matrices.append(matrix)
    print("Created matrix:", size)


    # 3.2 Функции измерения времени

def measure_cpu_time(operation):
    """Измеряет время выполнения операции на CPU в миллисекундах."""
    start_time = time.time()
    operation()
    end_time = time.time()

    return (end_time - start_time) * 1000


def measure_gpu_time(operation):
    """Измеряет время выполнения операции на GPU в миллисекундах."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start_event.record()
    operation()
    end_event.record()

    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)


# 3.3 Сравнение операций CPU и CUDA

def compare_operations(cpu_tensor):
    """Сравнивает скорость операций на CPU и GPU."""

    gpu_tensor = cpu_tensor.to("cuda")

    operations_cpu = {
        "Матричное умножение": lambda: torch.matmul(cpu_tensor, cpu_tensor.transpose(-1, -2)),
        "Сложение": lambda: cpu_tensor + cpu_tensor,
        "Умножение": lambda: cpu_tensor * cpu_tensor,
        "Транспонирование": lambda: cpu_tensor.transpose(-1, -2),
        "Сумма": lambda: torch.sum(cpu_tensor)
    }

    operations_gpu = {
        "Матричное умножение": lambda: torch.matmul(gpu_tensor, gpu_tensor.transpose(-1, -2)),
        "Сложение": lambda: gpu_tensor + gpu_tensor,
        "Умножение": lambda: gpu_tensor * gpu_tensor,
        "Транспонирование": lambda: gpu_tensor.transpose(-1, -2),
        "Сумма": lambda: torch.sum(gpu_tensor)
    }

    print("\nОперация                  | CPU (мс) | GPU (мс) | Ускорение")
    print("-" * 65)

    for name in operations_cpu:
        cpu_time = measure_cpu_time(operations_cpu[name])
        gpu_time = measure_gpu_time(operations_gpu[name])

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"{name:<25} | {cpu_time:8.3f} | {gpu_time:8.3f} | {speedup:8.2f}x")


# Выполняем сравнение для каждой матрицы
if torch.cuda.is_available():
    for matrix in matrices:
        print(f"\nРазмер матрицы: {tuple(matrix.shape)}")
        compare_operations(matrix)
else:
    print("\nCUDA недоступна, сравнение CPU и GPU невозможно.")


    # 3.4 Анализ результатов

"""
Анализ результатов:

1. Наибольшее ускорение на GPU наблюдается при операциях,
   которые хорошо распараллеливаются, например при матричном
   умножении и поэлементных операциях.

2. Некоторые простые операции могут выполняться на GPU медленнее.
   Это связано с накладными расходами на запуск CUDA-операций
   и синхронизацию CPU и GPU.

3. Размер и форма матриц влияют на ускорение. На достаточно больших
   задачах GPU обычно эффективнее, так как может выполнять большое
   количество вычислений параллельно.

4. Передача данных между CPU и GPU требует дополнительного времени.
   Поэтому частое копирование тензоров между устройствами может
   уменьшить преимущество GPU.

В полученных результатах максимальное ускорение составило около 14.19x
для операции сложения на тензоре размером 128x512x512.
Матричное умножение для этого тензора ускорилось примерно в 10.32 раза.

Операция транспонирования показала практически нулевое время на CPU,
поскольку torch.transpose обычно создает представление (view) тензора
с измененными strides, а не копирует все данные.
"""