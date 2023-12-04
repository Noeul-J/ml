"""
2023-02-13 K-NN 알고리즘
K-NN : K-Nearest Neighbor. One of Classification Algorithm
거리를 측정할 때는 유클리드 거리(Euclidean distance) 사용
K의 값에 따라 분류가 달라질 수 있다는 특징

도미와 빙어 데이터를 구분해보자
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier      # K-NN 알고리즘 클래스 KNeighborsClassifier import
import numpy as np

# data 1 - 도미 35마리 # 생선의 길이가 길수록 무게가 많이 나갈 확률이 높기 때문에 선형적인 결과를 나타냄
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# data 2 - 빙어 14마리
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# plt.scatter(bream_length, bream_weight)     # 선형도를 그리는 함수
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 도미와 빙어 데이터 구분
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]    # zip 함수는 나열된 리스트 각각에서 원소를 하나씩 꺼내 반환

# 정답 데이터 - 도미를 1로, 빙어를 0으로 나타냄
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier(n_neighbors=5)                 # k의 기본값이 5로 설정되어 있음. n_neighbors 매개 변수로 변경 가능.
kn.fit(fish_data, fish_target)                           # 도미를 찾기 위한 기준을 학습시킴
print(kn.score(fish_data, fish_target))                  # 훈련이 잘 되어 있는지 평가(0~1 사이의 값을 반환하고, 1은 모든 데이터를 정확히 맞혔단느 의미, 0.5는 절반의 데이터를 맞혔다는 의미)

# 훈련 셋과 데이터 셋 분리
input_arr = np.array(fish_data)                          # 파이썬 리스트를 넘파이 배열로 변환
target_arr = np.array(fish_target)

np.random.seed(42)                                       # random 값을 추출하기 위해 랜덤 시드를 설정
index = np.arange(49) # [0, 1, 2, ... , 48]
np.random.shuffle(index)

train_input = input_arr[index[:35]]                       # index[0] ~ index[34]의 인덱스에 대한 값을 한번에 배열로 반환
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 훈련 셋으로 학습시키고, 테스트 셋을 예측
kn.fit(train_input, train_target)
print(kn.score(train_input, train_target))

print(kn.predict(test_input))                            # [0 0 1 0 1 1 1 0 1 1 0 1 1 0]
print(test_target)                                       # [0 0 1 0 1 1 1 0 1 1 0 1 1 0]

# 주변 k개의 샘플을 알아보자
# distances, indexes = kn.kneighbors([[25, 150]])
#
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25,150, marker='^')
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# plt.xlim(0, 1000)                           # x축은 범위가 좁고, y축은 범위가 넓기 때문에 정확한 데이터가 안나옴. x축 범위를 y축과 동일하게 표현
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# print(distances)                            # [[ 92.00086956 130.73859415 137.17988191 138.32150953 138.39320793]]

# z-점수 표준화로 데이터 전처리
# 변수 X의 범위를 평균으로부터 몇 표준편차만큼 떨어져 있는지를 관점으로 변수를 확대/축소 함
# 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 낸 것이 분산이고, 표준편차는 분산의 제곱근이다.


mean = np.mean(train_input, axis=0)           # axis = 0 은 행을 따라 각 열의 통계 값을, axis = 1은 열을 따라 각 행의 통계 값을 계산한다.
std = np.std(train_input, axis=0)
print(mean, std)

train_scaled = (train_input - mean) / std     # 모든 행에서 mean에 있는 두 평균값을 빼주고, std에 있는 두 표준편차를 나누어준다.(브로드캐스팅 Broadcasting)
new = ([25, 150] - mean) / std

# x축과 y축의 범위가 모두 동일하게 바뀌었고, 데이터셋을 통해 다시 훈련해라
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
print(kn.predict([new]))

# scatter plot = 산점도
# 모든 행에 대해서 첫번째 열의 정보를 가져다 달라 - 가로 a[:,0]
# 모든 행에 대해서 두번째 열의 정보를 가져다 달라 - 세로 a[:,1]
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
