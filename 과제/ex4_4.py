import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# img는 MNIST 한 줄(row), label + 784 형태
def show(img): 
    # 픽셀 부분만 잘라서 (28x28)로 reshape 후 시각화
    plt.imshow(img[1:].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

# 0-9를 one-hot encoding 형태로 변환
def digit_to_vec(value):
    vec = np.zeros(10)
    vec[int(value)] = 1
    return vec

# CSV 파일에서 train 데이터 불러옴, 로컬 환경 동작
df_train = pd.read_csv("E:/대학교/4학년 1학기/실무데이터분석I/과제/과제자료/mnist_train.csv")
df_train.columns = ['label'] + [f'pxl{i}' for i in range(784)]

A_train = df_train.iloc[:, 1:].values  # shape: (N, 784)
B_train = np.array([digit_to_vec(v) for v in df_train['label'].values])  # shape: (N, 10)

# Least squares solution: X = (A^T A)^(-1) A^T B
# 유사 역행렬로 위 공식 계산
X = np.linalg.pinv(A_train) @ B_train

# test 데이터도 동일하게 로컬 환경에서 불러옴
df_test = pd.read_csv("E:/대학교/4학년 1학기/실무데이터분석I/과제/과제자료/mnist_test.csv")
df_test.columns = ['label'] + [f'pxl{i}' for i in range(784)]

A_test = df_test.iloc[:, 1:].values
B_test = np.array([digit_to_vec(v) for v in df_test['label'].values])  # shape: (M, 10)

# 각 이미지에 대해 예측 후 round 함수 적용
# 찾아보니까 one-hot처럼 예측할 경우에만 맞음
# 실제로는 argmax 쓰는게 보통
pred_scores = A_test @ X
rounded_pred = np.round(pred_scores)

# accuracy 계산
# 예측이 정답과 완전히 일치하는 경우만 count
correct = np.sum(np.all(rounded_pred == B_test, axis=1))
accuracy = correct / len(B_test)
print(f"Accuracy (rounded vector match): {accuracy:.2f}")