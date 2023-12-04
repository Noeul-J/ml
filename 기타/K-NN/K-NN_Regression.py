"""
2023-02-16 K-NN 회귀

회귀 : 임의의 숫자를 예측, 두 변수 사이의 상관관계를 분석
K-NN regression(K-최근접 이웃 회귀) : 주변의 가장 가까운 K개의 샘플을 통해 값을 예측

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
from sklearn.neighbors import KNeighborsRegressor

# [데이터 준비]
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# plt.scatter(perch_length, perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# [훈련 세트와 테스트 세트 나누기]
# 사이킷런에서 사용할 훈련 세트는 2차원 배열이어야 함
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# [회귀 모델 훈련]
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

print(knr.score(test_input, test_target)) # 0.992809406101064 -- 결정계수(Coefficient of determination), R^2

# R^2 = 1 - (타깃-예측)^2 / (타깃-평균)^2
# 타깃의 평균 정도를 예측하는 수준이라면 R^2는 0에 가까워지고, 타깃이 예측에 아주 가까워지면 1에 가까운 값이 된다.

print(knr.score(train_input, train_target)) # 0.9698823289099254

# 훈련 셋보다 테스트 셋의 점수가 높거나 두 점수가 모두 너무 낮은 경우 - 과소적합(Underfitting)
# 훈련 셋의 점수보다 테스트 셋의 점수가 지나치게 낮은 경우 - 과대적합(Overfitting)
# 과소적합 문제는 모델을 조금 더 복잡하게 만들면 된다. K-NN에서는 K값을 줄임으로써 모델을 복잡하게 만들수 있다.

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) # 0.9804899950518966
print(knr.score(test_input, test_target)) # 0.9746459963987609

# 길이가 100인 농어의 무게 예측
print(knr.predict([[100]])) # [1033.33333333]

# 테스트하고자 하는 샘플에 근접한 훈련 데이터가 없는 경우, 즉 훈련 셋의 범위를 많이 벗어나는 샘플인 경우 정확하게 예측하기 어려운 한계가 있다

