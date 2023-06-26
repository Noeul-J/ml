import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine_csv_data')

print(wine.head())      # 알코올 도수, 당도, pH -- 처음 5개의 샘플 확인
print(wine.info())      # 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는데 용이
                        # 누락된 값이 있는 경우, 그 데이터를 버리거나 평균값으로 채운 후 사용

print(wine.describe())  # 평균, 표준편차, 최소, 최대 값 확인 가능

# 훈련 세트와 테스트 세트 나누기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# 샘플 개수 20% 정도를 테스트 세트로 나눠
print(train_input.shape, test_input.shape)

# 표준점수로 변환
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

# # 트리 알고리즘
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_scaled, train_target)
# print(dt.score(train_scaled, train_target))             # 훈련 세트
# print(dt.score(test_scaled, test_target))               # 테스트 세트
#
# plt.figure(figsize=(10, 7))
# plot_tree(dt)
# plt.show()
#
# # 트리의 깊이를 제한해서 출력
# plt.figure(figsize=(10, 7))
# plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# # max_depth=1 >> 루트 노드를 제외하고 하나의 노드를 더 확장해서 그려
# # filled >> 클래스에 맞게 노드의 색을 칠함
# plt.show()

# gini는 지니 불순도를 의미(Gini impurity)
# criterion 매개변수의 기본값이 gini - 노드에서 데이터를 분할할 기준을 정하는 것

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# plt.show()

print(dt.feature_importances_)                          # [0.12345626 0.86862934 0.0079144 ]
