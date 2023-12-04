import numpy as np
from sklearn.neighbors import BallTree

X = [[1,1550], [900,440], [2500,330], [4000,2], [5000,1]]
X = np.asarray(X)

# 트리 생성
tree = BallTree(X)

# 테스트 데이터 쿼리
dist, ind = tree.query([[1, 1551]], 1)

print(dist, ind)