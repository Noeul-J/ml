import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn 데이터셋에서 iris 데이터셋 로딩
from sklearn import datasets

# Train-Test 데이터셋 분할
from sklearn.model_selection import train_test_split

#K-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# SVM
from sklearn.svm import SVC

# 로지스틱회귀
from sklearn.linear_model import LogisticRegression

# 데이터 가져오기
iris = datasets.load_iris()

# # key값 확인
# print(iris.keys())

# # 데이터셋에 대한 설명
# print(iris['DESCR'])

# # target 속성
# print('데이터셋 크기: ', iris['target'].shape)
# print('데이터셋 내용: ', iris['target'])

# # data 속성
# print('데이터셋 크기: ', iris['data'].shape)
# print('데이터셋 내용: ', iris['data'][:7, :])   # 맨 위 7개 행 출력

# 데이터프레임 변환
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
# print(" 데이터프레임의 형태 ", df.shape)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['Target'] = iris['target']       # Target 열 추가
print(df.head())

# 데이터프레임의 기본 정보를 보여줌
print(df.info)  # 유효값(non-null) : 결측값이 아닌
# 통계 정보 요약
print(df.describe())    # 평균값, 표준편차, 최소값, 최대값
# 결측값 확인
print(df.isnull().sum())
# 중복 데이터 확인
print(df.duplicated().sum())
print(df.loc[df.duplicated(),:])    # 어느 행의 데이터가 중복인지 확인 가능
print(df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :])  # 어떤 데이터와 중복인지 확인

df = df.drop_duplicates()   #중복 데이터 삭제
print(df.loc[(df.sepal_length==5.8) & (df.petal_width==1.9), :])   

# 상관관계 분석
print(df.corr())    # 변수 간의 상관계수 행렬을 출력

# 데이터 시각화
sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

# target 값의 분포
print(df['Target'].value_counts())      # 데이터 종류별 샘플 개수 출력
# plt.hist(x='sepal_length', data=df)     # 데이터 프레임을 지정하고, x 옵션에 열이름을 입력 > 히스토그램
# sns.displot(x='sepal_width', kind='hist', data=df)

# KDE 밀도함수
# sns.displot(x='petal_width', kind='kde' , data=df)  # 두개의 봉오리 형태 > 서로 다른 이질적인 데이터가 섞여있다
# 품좀별 분포를 그리려면 hue 옶션을 주면 돼
# sns.displot(x='petal_width', hue='Target', kind='kde', data=df)

# pairplot() 이용해서 서로 다른 피처 간 관계를 나타내는 그래프 생성
# sns.pairplot(df, hue='Target', height=2.5, diag_kind='kde')
# plt.show()

# Train-Test 데이터셋 분할
x_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                    y_data,
                                                    test_size=0.2,   # 전체 중 20%를 테스트용으로 분할, 나머지는 훈련용
                                                    shuffle=True,   # 무작위로 섞어서 추출
                                                    random_state=20)    # 무작위 추출 시 일정한 기준으로 분할 - 코드를 다시 실행해도 같은 결과를 얻음
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# K-NN 알고리즘. 데이터 x 가 주어지면 기존 데이터 중 속성이 비슷한 k개의 이웃을 먼저 찾는 알고리즘
knn = KNeighborsClassifier(n_neighbors=7)   # k값을 7로 설정
knn.fit(x_train, y_train)
y_knn_pred = knn.predict(x_test)
print("예측값: ", y_knn_pred[:5])
knn_acc = accuracy_score(y_test, y_knn_pred)
print("Accuracy: %.4f" % knn_acc)

# SVM(Support Vector Machine)은 데이터셋의 각 피처(열) 벡터들이 고유의 축을 갖는 벡터 공간을 이룬다고 가정한다.
# 모든 데이터를 벡터 공간 내의 좌표에 점으로 표시하고, 각 데이터가 속하는 목표 클래스별로 군집을 이룬다고 생각한다
svc = SVC(kernel='rbf')  # 'rbf'는 Radial Basis Foundation을 뜻한다.
svc.fit(x_train, y_train)
y_svc_pred = svc.predict(x_test)
print("예측값: ", y_svc_pred[:5])

svc_acc = accuracy_score(y_test, y_svc_pred)
print("Accuracy: %.4f" % svc_acc)

# 로지스틱 회귀. 시그모이드 함수의 출력값(0~1사이)을 각 분류 클래스에 속하게 될 확률값으로 사용
# 1에 가까우면 해당 클래스로 분류하고, 0에 가까우면 아니라고 분류햔다.
lrc = LogisticRegression()
lrc.fit(x_train, y_train)
y_lrc_pred = lrc.predict(x_test)
print("예측값: ", y_lrc_pred[:5])
lrc_acc = accuracy_score(y_test, y_lrc_pred)
print("Accuracy: %4.f" % lrc_acc)
y_lrc_prob=lrc.predict_log_proba(x_test)
print(y_lrc_prob)