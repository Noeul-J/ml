import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt

fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target , test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# SGDClassifier 의 객체 생성 시 2개의 매개변수 지정
# loss - 손실 함수의 종류 지정. max_iter는 수행할 에포크 횟수

# sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
# sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))           # 0.773109243697479  / 0.775
#
# # sc를 추가로 더 훈련
# sc.partial_fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))          #  0.8151260504201681  / 0.85

# 에포크가 적은 모델은 과소적합될 확률이, 높은 모델은 과대 적합될 확률이 높다.
# 과대적합 되기 전에 훈련을 멈추는 것 - 조기 종료(early stopping)
# sc = SGDClassifier(loss='log', random_state=42)
# train_score = []
# test_score = []
# classes = np.unique(train_target)               # train_taret에 있는 7개 생선의 목록을 만들어
#
# for _ in range(0, 300):
#     sc.partial_fit(train_scaled, train_target, classes = classes)
#     train_score.append(sc.score(train_scaled, train_target))
#     test_score.append(sc.score(test_scaled, test_target))
#
# plt.plot(train_score)
# plt.plot(test_score)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()                            # 100 에포크가 적당해 보임

# sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
# sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))                       #0.957983193277311   / 0.925

# loss 매개변수의 기본값은 ‘hinge’임. hinge loss 는 support vector machine이라 불리는 또 다른 머신러닝 알고리즘을 위한 손실 함수
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))