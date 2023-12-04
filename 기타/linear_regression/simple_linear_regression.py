"""
2023-03-25 선형 회귀

선형 회귀(Linear Regression)는 널리 사용되는 대표적인 회귀 알고리즘이다.
선형 회귀는 종속 변수 y와 하나 이상의 독립 변수 x와의 선형 상관관계를 모델링하는 기법이다.
만약 독립 변수 x가 1개라면 단순 선형 회귀라고 하고, 2개 이상이면 다중 선형 회귀라고 한다.

1) 단순 선형 회귀 (Simple Linear Regression)
단순 선형 회귀는 y = Wx+b의 식으로 나타난다. 머신러닝에서는 독립 변수 x에 곱해지는 W값을 가중치(weight), 상수항에 해당하는 b를 편향(bias)이라고 부른다.
따라서 단순 선형 회귀 모델을 훈련하는 것은 적절한 W와 b값을 찾는 것이다. 그래프의 형태는 직선으로 나타난다.


2) 다중 선형 회귀 (Multiple Linear Regression)
다중 선형 회귀는 y = W1x1+W2x2+...+Wnxn+b 의 식으로 나타난다.
여러 독립 변수에 의해 영향을 받는 경우이다. 만약 2개의 독립 변수면 그래프는 평면으로 나타날 것이다.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

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

# [훈련 세트와 테스트 세트 나누기]
# 사이킷런에서 사용할 훈련 세트는 2차원 배열이어야 함
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 선형회귀 클래스
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[100]])) # [3192.69585141]

# y=ax+b 에서 a는 lr.coef, b는 lr.intercept에 저장됨
print(lr.coef_, lr.intercept_) # [39.01714496] -709.018644953547  ---  y=(39.01...)x−(709.0186...)

# 직선을 그래프로 시각화
plt.scatter(train_input, train_target)
plt.plot([15, 100], [15*lr.coef_ + lr.intercept_, 100*lr.coef_ + lr.intercept_])
plt.scatter(100, 3192, marker='^')
# plt.show()

# 그래프가 직선이기 때문에 길이가 10 이하로 내려가면 무게를 음수로 예측하게 되는 것을 그래프에서 볼 수 있는데, 이는 실제로 일어날 수 없는 일이다.
# 훈련한 모델에 대한 R^2 점수 또한, 훈련 셋과 테스트 셋의 점수에 차이가 있을 뿐더러 훈련 셋의 점수도 그렇게 높지 않다.
print(lr.score(train_input, train_target)) # 0.9398463339976041
print(lr.score(test_input, test_target)) # 0.8247503123313562

# 그 이유는 실제로 농어의 데이터는 직선보단 곡선에 가깝기 때문이다
# 2차 방정식 구하기
# column_stack 함수는 열 방향으로 배열을 합쳐주는 기능을 제공한다
# 1열에는 길이의 제곱, 2열에는 길이가 들어있는 2차원 배열이 train_poly와 test_poly에 할당된다.
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))

# 선형 회귀 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.coef_, lr.intercept_) # [  1.01433211 -21.55792498] 116.05021078278259
# 이차항의 계수가 1.014..., 일차항의 계수가 -21.5579... 를 의미한다.
# y = 1.014x^2 -21.5579x + 116.05   - 다항 회귀(Polynomial Regression)
# 다중 회귀는 2개 이상의 독립 변수가 존재하는 형태이고, 다항 회귀는 하나의 독립 변수를 이용하여 차수를 높이는 개념이라고 생각하면 된다.

print(lr.predict([[100**2, 100]])) # [8103.57880667]

print(lr.score(train_poly, train_target)) # 0.9706807451768623
print(lr.score(test_poly, test_target)) # 0.9775935108325121

point = np.arange(15, 100)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point +  116.05)
plt.scatter([100], [8103], marker='^')
plt.show()
