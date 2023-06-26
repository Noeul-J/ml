import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax
"""
2023-04-13

로지스틱 회귀(Logistic Regression)
- 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서 1사이의 값으로 예측하고
그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 뷴류해주는 지도 학습 알고리즘

2진분류(binary classification)

- 모든 속성(feature)들의 계수(coefficient)와 절편(intercept)을 0으로 초기화한다
- 각 속성들의 값(value)에 계수(coefficient)를 곱해서 log-odds를 구한다.
- log-odds를 sigmoid 함수에 넣어서 [0,1] 범위의 확률을 구한다.

odds = 사건이 발생할 확률 / 사건이 발생하지 않을 확률

"""

# 데이터 준비
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
print(fish.head())
print(pd.unique(fish['Species']))

# 입력 데이터와 타겟 데이터
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()

# 훈련 세트와 테스트 세트를 나눠
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기의 확률 예측
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)      # 타깃값을 사이킷런 모델에 전달하면 순서가 알파벳순으로 자동 재배열됨
print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))              # 소수점 네번째 자리까지 표시. 다섯번째에서 반올림

# 시그모이드 함수
# z = np.arange(-5, 5, 0.1)
# phi = 1 / (1 + np.exp(-z))
# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

# 불리언 인덱싱 - True, False로 원하는 값 전달
# char_arr = np.array(['A', 'B', 'C', 'D'])
# print(char_arr[[True, False, True, False]])   #['A','C']

# 불리언 인덱싱을 이용해 도미와 빙어 데이터만 골라내
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')  # 도미와 빙어만 True
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.classes_)
print(lr.predict(train_bream_smelt[:5]))             # 예측
print(lr.predict_proba(train_bream_smelt))          # 확률 (빙어가 양성 1)

# 로지스틱 희귀가 학습한 계수
# z = a x (Weight) + b x (Length) + c x (Diagonal) + d x (Height) + e x (Width) + f
print(lr.coef_, lr.intercept_)      # [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# z = -0.404 x (weight) -0.576 x (Length) -0.663 x (Diagonal) -1.013 x (Height) - 0.732 x (Width) - 2.161

decision = lr.decision_function(train_bream_smelt[:5])  # 처음 5개 샘플의 z 값
print(decision)
print(expit(decision))

# 반복횟수, 매개변수 조절
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])           # 테스트 세트의 처음 5개 샘플에 대한 예측 확률 출력
print(np.round(proba, decimals=3))                  # 소수점 네 번째 자리ㅔ서 반올림
"""
[[0.    0.014 0.841 0.    0.136 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.935 0.015 0.016 0.   ]
 [0.011 0.034 0.306 0.007 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
 """

print(lr.classes_)                  # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

print(lr.coef_.shape, lr.intercept_.shape)       # 클래스 마다 z값을 하나씩 계산 -> 확률은 softmax함수 사용
                                                # 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축하고 전체 합이 1이 되도록 만든다
                                                # 정규화된 지수 함수라고 부르기도 함

# e_sum = e^z1+e^z2=e^z3+e^z4=e^z5+e^z6+e^z7

decision = lr.decision_function(test_scaled[:5]) # z1~z7 값을 구해
print(np.round(decision, decimals=2))

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))