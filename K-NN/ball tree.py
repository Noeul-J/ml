# https://www.astroml.org/book_figures/chapter2/fig_balltree_example.html
# https://en.wikipedia.org/wiki/Ball_tree
import numpy as np

class Node:
    def __init__(self, data, radius, depth, left_child=None, right_child=None):
        self.left_child = left_child
        self.right_child = right_child
        self.data = data
        self.radius = radius
        self.depth = depth

    def printALL(self):
        print(self.radius, self.data, self.depth)
        if self.left_child != None:
            self.left_child.printALL()
        if self.right_child != None:
            self.right_child.printALL()

def balltree(ndata, depth):
    if ndata.shape[0] < 1:
        return None

    # element가 한 개일 경우
    if ndata.shape[0] == 1:
        return Node(
            data=np.max(ndata, 0).tolist(),
            radius=0,
            depth=depth,
            left_child=None,
            right_child=None
        )
    else:
        # 범위가 가장 큰 dimension에 따라 정렬
        largest_dim = np.argmax(ndata.max(0) - ndata.min(0))
        i_sort = np.argsort(ndata[:, largest_dim])
        ndata[:] = ndata[i_sort, :]

        nHalf = int(ndata.shape[0] / 2)
        loc = ndata[nHalf, :]
        data = loc.tolist()

        # 중간 값(data)에서 가장 멀리 떨어진 값 까지의 거리
        radius = np.sqrt(np.max(np.sum((ndata - loc) ** 2, 1)))

        return Node(
            data=data,
            radius=radius,
            depth=depth,
            left_child=balltree(ndata[:nHalf], depth+1),
            right_child=balltree(ndata[nHalf+1:], depth+1)
        )

if __name__ == '__main__':
    X = [[1,1550], [900,440], [2500,330], [4000,2], [5000,1]]
    X = np.asarray(X)

    tree = balltree(X, 0)
    tree.printALL()