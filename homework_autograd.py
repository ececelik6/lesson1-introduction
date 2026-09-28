import torch


# 2.1 Простые вычисления с градиентами

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# Проверка типов
if not all(isinstance(value, torch.Tensor) for value in (x, y, z)):
    raise TypeError("x, y и z должны быть тензорами PyTorch.")

if not all(value.requires_grad for value in (x, y, z)):
    raise ValueError("Для x, y и z должен быть установлен requires_grad=True.")

# f(x, y, z) = x^2 + y^2 + z^2 + 2xyz
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Вычисляем градиенты
f.backward()

print("f =", f.item())
print("df/dx =", x.grad.item())
print("df/dy =", y.grad.item())
print("df/dz =", z.grad.item())

# Аналитическая проверка:
# df/dx = 2x + 2yz
# df/dy = 2y + 2xz
# df/dz = 2z + 2xy
analytical_dx = 2 * x.item() + 2 * y.item() * z.item()
analytical_dy = 2 * y.item() + 2 * x.item() * z.item()
analytical_dz = 2 * z.item() + 2 * x.item() * y.item()

print("Analytical df/dx =", analytical_dx)
print("Analytical df/dy =", analytical_dy)
print("Analytical df/dz =", analytical_dz)

# Проверяем совпадение градиентов
assert torch.isclose(x.grad, torch.tensor(analytical_dx))
assert torch.isclose(y.grad, torch.tensor(analytical_dy))
assert torch.isclose(z.grad, torch.tensor(analytical_dz))


# 2.2 Градиент функции потерь

def mse_loss(y_pred, y_true):
    """Вычисляет среднеквадратичную ошибку (MSE)."""
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("y_pred и y_true должны быть тензорами PyTorch.")

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred и y_true должны иметь одинаковую форму.")

    if not (y_pred.is_floating_point() and y_true.is_floating_point()):
        raise TypeError("y_pred и y_true должны иметь тип с плавающей точкой.")

    return torch.mean((y_pred - y_true) ** 2)


# Простые тестовые данные
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# y_pred = w * x + b
y_pred = w * x_data + b

loss = mse_loss(y_pred, y_true)

# Вычисляем градиенты по w и b
loss.backward()

print("\nMSE =", loss.item())
print("Gradient w =", w.grad.item())
print("Gradient b =", b.grad.item())

# Проверка на простом примере
expected_loss = torch.tensor(14.0 / 3.0)
expected_w_grad = torch.tensor(-28.0 / 3.0)
expected_b_grad = torch.tensor(-4.0)

assert torch.isclose(loss.detach(), expected_loss)
assert torch.isclose(w.grad, expected_w_grad)
assert torch.isclose(b.grad, expected_b_grad)


# 2.3 Цепное правило

x_chain = torch.tensor(2.0, requires_grad=True)

# f(x) = sin(x^2 + 1)
f_chain = torch.sin(x_chain**2 + 1)

# Градиент с помощью backward()
f_chain.backward()
gradient_backward = x_chain.grad.detach().clone()

print("\nf(x) =", f_chain.item())
print("df/dx using backward =", gradient_backward.item())

# Проверка с помощью torch.autograd.grad
x_check = torch.tensor(2.0, requires_grad=True)
f_check = torch.sin(x_check**2 + 1)

gradient_autograd = torch.autograd.grad(f_check, x_check)[0]

print("df/dx using torch.autograd.grad =", gradient_autograd.item())

# Аналитическая проверка:
# df/dx = cos(x^2 + 1) * 2x
analytical_gradient = torch.cos(x_check**2 + 1) * 2 * x_check

print("Analytical df/dx =", analytical_gradient.item())

assert torch.isclose(gradient_backward, gradient_autograd)
assert torch.isclose(gradient_autograd, analytical_gradient)

print("\nВсе проверки homework_autograd.py успешно пройдены.")