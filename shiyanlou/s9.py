import numpy as np 
from matplotlib import pyplot as plt 
import numpy as np 
import operator

def create_data():
    """生成示例数据
    """

    features = np.array(
        [[2.88, 3.05], [3.1, 2.45], [3.05, 2.8], [2.9, 2.7], [2.75, 3.4],
         [3.23, 2.9], [3.2, 3.75], [3.5, 2.9], [3.65, 3.6],[3.35, 3.3]])
    labels = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    return features, labels

def d_man(x, y):
    """曼哈顿距离
    """
    return np.abs(x-y).sum()

def d_euc(x, y):
    """欧式距离
    """
    return np.sqrt(np.square(x - y).sum())
def majority_voting(class_count):
    """排序"""
    return sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

def knn_classify(test_data, train_data, labels, k):
    """knn算法"""
    distances = np.array([])  # 距离

    for each_data in train_data:
        d = d_euc(test_data, each_data)
        distances = np.append(distances, d)

    sorted_distance_index = distances.argsort()
    sorted_distance = np.sort(distances)
    r = (sorted_distance[k] + sorted_distance[k-1]) / 2

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    final_label = majority_voting(class_count)
    return final_label, r


if __name__ == '__main__':
    # 查看示例数据
    features, labels = create_data()
    # print('features: \n', features)
    # print('labels: \n', labels)

    # 示例数据绘图
    # plt.figure(figsize=(5, 5))
    # plt.xlim((2.4, 3.8))    
    # plt.ylim((2.4, 3.8))    
    # x_feature = list(map(lambda x:x[0], features))
    # y_feature = list(map(lambda x:x[1], features))
    # plt.scatter(x_feature[:5], y_feature[:5], c='b')
    # plt.scatter(x_feature[5:], y_feature[:5], c='g')
    # plt.scatter([3.18], [3.15], c='r', marker='x')
    # plt.show()
    
    # 测试曼哈顿距离
    # x = np.array([3.1, 3.2])
    # print("x:", x)
    # y = np.array([2.5, 2.8])
    # print("y:", y)
    # d_man = d_man(x, y)
    # print(d_man)

    # 测试欧式距离
    # x = np.random.random(10)  # 随机生成10个数的数组作为x特征的值
    # print("x:", x)
    # y = np.random.random(10)
    # print("y:", y)
    # distance_euc = d_euc(x, y)
    # print(distance_euc)
    
    # 测试排序函数
    # arr = {'A': 3, 'B': 2, 'C': 6, 'D':5, 'ZSH':100}
    # print(majority_voting(arr))    
    
    # 测试knn算法
    test_data = np.array([3.18, 3.15])
    final_label, r =  knn_classify(test_data, features, labels, 5)
    print(final_label)