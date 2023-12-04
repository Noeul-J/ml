from scipy import spatial

if __name__ == '__main__':
    point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
    tree = spatial.KDTree(point_list)

    print(tree.query((2, 3)))
    print(tree.query((2, 3.5)))