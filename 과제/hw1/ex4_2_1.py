import numpy as np

x = np.array(range(1, 25))
y = np.array([
    75, 77, 76, 73, 69, 68, 63, 59,
    57, 55, 54, 52, 50, 50, 49, 49,
    49, 50, 54, 56, 59, 63, 67, 72
])

# 행렬 X: [x^2, x, 1]
X = np.vstack([x**2, x, np.ones_like(x)]).T

#  beta = (X^T X)^-1 X^T y
beta = np.linalg.inv(X.T @ X) @ X.T @ y
A, B, C = beta
print(f"A = {A:.2f}, B = {B:.2f}, C = {C:.2f}")

# 예측값 f(x)
y_pred = A * x**2 + B * x + C

# MSE
mse = np.mean((y - y_pred)**2)
print(f"MSE (E2 Error) = {mse:.2f}")