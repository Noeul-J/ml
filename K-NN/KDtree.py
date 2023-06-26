from operator import itemgetter

class Node:
    def __init__(self, data, depth, left_child=None, right_child=None):
        self.left_child = left_child
        self.right_child = right_child
        self.data = data
        self.depth = depth
    def printALL(self):
        print(self.depth, self.data)
        if self.left_child != None:
            self.left_child.printALL()
        if self.right_child != None:
            self.right_child.printALL()


def kdtree(point_list, depth=0):
    if not point_list:
        return None

    k = len(point_list[0])
    axis = depth % k

    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2

    return Node(
        data=point_list[median],
        depth=depth,
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )

if __name__ == '__main__':
    point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
    tree = kdtree(point_list)
    tree.printALL()