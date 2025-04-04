import numpy as np
import matplotlib.pyplot as plt

# 주어진 온도 데이터
raw_data = [
    (1, 75), (2, 77), (3, 76), (4, 73), (5, 69), (6, 68),
    (7, 63), (8, 59), (9, 57), (10, 55), (11, 54), (12, 52),
    (13, 50), (14, 50), (15, 49), (16, 49), (17, 49), (18, 50),
    (19, 54), (20, 56), (21, 59), (22, 63), (23, 67), (24, 72)
]

# 홀수시간: train, 짝수시간: test
train_data = [(x, y) for x, y in raw_data if x % 2 == 1]
test_data = [(x, y) for x, y in raw_data if x % 2 == 0]

x_train = np.array([x for x, y in train_data])
y_train = np.array([y for x, y in train_data])
x_test = np.array([x for x, y in test_data])
y_test = np.array([y for x, y in test_data])

# 입력 x랑 degree 받아서 행렬 생성
def design_matrix(x, degree):
    return np.vstack([x**d for d in range(degree + 1)]).T  # shape: (n_samples, degree+1)

# LSS
def lss_fit(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Ridge, 람다는 0.1 고정
def ridge_fit(X, y, lambd):
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lambd * np.identity(n)) @ X.T @ y

# 예측 함수
def predict(X, coef):
    return X @ coef

# E2 오차, MSE 계산
def compute_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 결과 테이블 초기화
lss_table = []
ridge_table = []

for p in range(1, 11):
    X_train = design_matrix(x_train, p)
    X_test = design_matrix(x_test, p)
    
    # LSS 테이블에 E2 저장
    coef_lss = lss_fit(X_train, y_train)
    y_train_pred = predict(X_train, coef_lss)
    y_test_pred = predict(X_test, coef_lss)
    lss_train_error = compute_error(y_train, y_train_pred)
    lss_test_error = compute_error(y_test, y_test_pred)
    lss_table.append((lss_train_error, lss_test_error))
    
    # Ridge 테이블에 E2 저장
    coef_ridge = ridge_fit(X_train, y_train, lambd=0.1)
    y_train_pred_ridge = predict(X_train, coef_ridge)
    y_test_pred_ridge = predict(X_test, coef_ridge)
    ridge_train_error = compute_error(y_train, y_train_pred_ridge)
    ridge_test_error = compute_error(y_test, y_test_pred_ridge)
    ridge_table.append((ridge_train_error, ridge_test_error))

# 결과 출력
print("LSS Table (E2_train, E2_test)")
for i, (e_train, e_test) in enumerate(lss_table, 1):
    print(f"Degree {i}: {e_train:.4f}, {e_test:.4f}")

print("\nRidge Table (E2_train, E2_test)")
for i, (e_train, e_test) in enumerate(ridge_table, 1):
    print(f"Degree {i}: {e_train:.4f}, {e_test:.4f}")
