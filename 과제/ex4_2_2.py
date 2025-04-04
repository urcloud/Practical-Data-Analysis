import numpy as np
import matplotlib.pyplot as plt

temps_raw = [
    "75at01", "77at02", "76at03", "73at04", "69at05", "68at06", "63at07", "59at08",
    "57at09", "55at10", "54at11", "52at12", "50at13", "50at14", "49at15", "49at16",
    "49at17", "50at18", "54at19", "56at20", "59at21", "63at22", "67at23", "72at24"
]

# 문자열 끝 두자리 = x배열, 문자열 앞 두자리 = y배열
x = np.array([int(s[-2:]) for s in temps_raw], dtype=float)
y = np.array([int(s[:-4]) for s in temps_raw], dtype=float)

# 실제값과 오차 제곱 후 평균 - MSE 반환
def E2sq(x, y, beta):
    A, B, C = beta
    y_hat = A * np.cos(B * x) + C
    return np.mean((y - y_hat) ** 2)

# Gradient
def grad_E2(x, y, beta):
    A, B, C = beta
    cosBx = np.cos(B * x)
    sinBx = np.sin(B * x)
    y_hat = A * cosBx + C
    # 실제-예측값
    residual = y - y_hat

    # 손실함수 편미분값, Gradient 벡터 구성
    grad_A = -2 * np.mean(residual * cosBx)
    grad_B =  2 * np.mean(residual * A * sinBx * x)
    grad_C = -2 * np.mean(residual)

    return np.array([grad_A, grad_B, grad_C])

# Hessian 행렬
def hessian_E2(x, y, beta):
    A, B, C = beta
    cosBx = np.cos(B * x)
    sinBx = np.sin(B * x)
    y_hat = A * cosBx + C
    residual = y - y_hat

    d2_AA = 2 * np.mean(cosBx ** 2)
    d2_AB = -2 * np.mean(x * sinBx * cosBx * A - residual * x * sinBx)
    d2_AC = 2 * np.mean(cosBx)

    d2_BB = 2 * np.mean((A * x * sinBx)**2 - residual * A * x**2 * cosBx)
    d2_BC = -2 * np.mean(A * x * sinBx)

    d2_CC = 2

    H = np.array([
        [d2_AA, d2_AB, d2_AC],
        [d2_AB, d2_BB, d2_BC],
        [d2_AC, d2_BC, d2_CC]
    ])
    return H

# 초기값 지정 (A, B, C)
beta = np.array([10.0, 0.2, 50.0])
beta_list = []

# 최대 40번 반복하며 최적화 반복
for k in range(40):
    grad = grad_E2(x, y, beta)
    hess = hessian_E2(x, y, beta)

    try:
        # 뉴턴방향 계산
        delta = np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        print(f"[{k}] Hessian not invertible!")
        break

    # 파라미터 갱신
    beta = beta - delta
    beta_list.append(beta.copy())

    # 현재 파라미터와 Loss값 출력
    print(f"{k:2d} | A: {beta[0]:.4f}, B: {beta[1]:.4f}, C: {beta[2]:.4f}, E2: {E2sq(x,y,beta):.6f}")

# 결과 시각화
x_dense = np.linspace(1, 24, 300)
y_fit = beta[0] * np.cos(beta[1] * x_dense) + beta[2]

plt.figure(figsize=(10, 5))
plt.plot(x, y, 'ro', label='Observed')
plt.plot(x_dense, y_fit, 'b-', label='Fitted curve')
plt.xlabel('Hour')
plt.ylabel('Temperature (°F)')
plt.title('Least Squares Fit: y = A cos(Bx) + C')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 최종 E2, A, B, C 파라미터, MSE 계산 및 출력
def E2Loss(x, y, beta):
    A, B, C = beta
    y_hat = A * np.cos(B * x) + C
    return np.mean((y - y_hat) ** 2) 

final_A, final_B, final_C = beta
final_E2 = E2sq(x, y, beta)

print("\n최종 파라미터 및 손실값")
print(f"A = {final_A:.2f}")
print(f"B = {final_B:.2f}")
print(f"C = {final_C:.2f}")
print(f"MSE = {E2Loss(x, y, beta):.2f}")