"""
2023-02-15 K-NN 알고리즘
K-NN : K-Nearest Neighbor. One of Classification Algorithm
거리를 측정할 때는 유클리드 거리(Euclidean distance) 사용
K의 값에 따라 분류가 달라질 수 있다는 특징

도미와 빙어 데이터를 구분해보자
"""

import numpy as np
from sklearn.model_selection import train_test_split            # 훈련 세트와 테스트 세트 나누기
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# [데이터]
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# [두 리스트 묶기]
# print(np.column_stack(([1,2,3], [4,5,6]))) -- 두 리스트를 차례로 묶어
fish_data = np.column_stack((fish_length, fish_weight))
# print(fish_data[:5])

# [target data]
# np.ones(n) 1이 n개인 데이터 생성
# np.zeros(n) 0이 n개인 데이터 생성
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# print(fish_target)

# [훈련 세트와 테스트 세트 나누기]
# random_state가 random.seed를 지정할 수 있는 매개변수
# 2개의 배열이 2개씩 나뉘어 총 4개의 배열이 반환된다.
# 훈련 데이터와 테스트 데이터는 각각 36, 13개로 나뉘어졌고, 입력데이터는 2개의 열이 있는 2차원 배열이고 타깃 데이터는 1차원 배열
# train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)
# print(test_target)      #[1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.] ->  샘플링 편향이 있음

# stratify 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눔
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# print(test_target)

# [K-최근접 이웃 훈련]
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
result = kn.score(test_input, test_target)
# print(result)                   # 1.0으로 분류를 정확히 했는데
# print(kn.predict([[25, 150]]))  # 왜 1(도미)이 아닐까

# [수상한 도미 그래프로 나타내기]
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker='^')        # marker 매개변수는 모양을 지정
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# [이웃 샘플 따로 구분해서 그래프 나타내기]
# distances, indexes = kn.kneighbors([[25, 150]])
#
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()
# print(train_target[indexes])    # [[1. 0. 0. 0. 0.]] 빙어가 더 많아
# print(distances)                # [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]

# [x축 범위 조정 해서 그래프 나타내기]
# x축의 범위는 10~40, y축의 범위는 0~1000 이므로  y축으로 조금만 멀어져도 거리가 큰값으로 계산됨
# 따라서 x축의 범위를 0~1000으로 맞춰보자
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
# plt.xlim((0, 1000))             # x축 범위를 지정
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# [데이터 전처리]
# 표준점수(standard score, =z점수) 사용
# 분산은 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구한다.
# 표준편차는 분산의 제곱근으로 데이터가 분산된 정도를 나타냄

# axis=0 -- 각 행을 따라 각 열의 통계값 계산
# axis=1 -- 각 열을 따라 각 행의 통계값 계산
mean = np.mean(train_input, axis=0)     # 함수의 평균을 계산
std = np.std(train_input, axis=0)       # 함수의 표준편차를 계산
# print(mean, std)

# 브로드캐스팅(broadcasting)
# train_input의 모든 행에서 mean에 있는 두 평균값을 빼준다.
train_scaled = (train_input - mean) / std       # 표준점수 계산

# [전처리 데이터로 모델 훈련하기]
new = ([25, 150] - mean) / std
# plt.scatter(train_scaled[:,0], train_scaled[:,1])
# plt.scatter(new[0], new[1], marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
print(kn.predict([new]))

# [결과 그래프로 나타내기]
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
