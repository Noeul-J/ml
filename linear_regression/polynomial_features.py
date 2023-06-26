import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


n = 100

# np.random.rand(n,1) n행 1열로 난수 발생(0~1 사이 값)
x = 6 * np.random.rand(n, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.rand(n, 1)

plt.scatter(x, y, s=5)
# 데이터가 곡선으로 나타남 > 다항 회귀를 사용
# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)        # 기본 다항식형태를 만듦  include_bias=True 일 경우 0차항(1)도 함께 만듦
x_poly = poly_features.fit_transform(x)                                 # 새롭게 정의된 numpy 배열은 행별로 각 데이터를 다항 형태로 변형해준다.
print(x[0], x_poly[0]) # [2.14778695] [2.14778695 4.61298876]

model = LinearRegression()
model.fit(x_poly, y)            # 다항회귀 모델에 변형한 데이터와 기존 y값을 넣고 학습시킨다

# 데이터 변환 과정과 머신러닝을 연결해주는 파이프라인
# make_pipeline을 통해 PolynomialFeatures 와 LinearRegression의 과정이 한번으로 통합된 모델을 생성
# 이렇게 생성된 모델은 step에 따라 과정이 나뉨(steps 함수도 존재하여 파이프라인에 포함된 모델의 형태를 확인 가능)
model_lr = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model_lr.fit(x, y)
print(model_lr.steps[1][1].coef_)    # [[0.97739498 0.52065734]]

# 다항회귀 그래프
xx = np.linspace(-3, 3, 100)
y_pred = model_lr.predict(xx[:, np.newaxis])
plt.plot(xx, y_pred)
plt.scatter(x, y, s=5)
plt.show()
