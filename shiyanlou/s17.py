import numpy as np 
from sklearn import datasets
from matplotlib import pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 


def euclidean_distance(a, b):
    """计算欧氏距离
    参数：
    a：
    b：
    返回：
    dist：
    """
    return np.sqrt(np.sum(np.square(a-b)))
    
def agglomerative_clustering(data):
    """Agglomerative聚类过程
    """
    while len(data) > 1:
        print("第{}次迭代".format(10 - len(data) + 1))
        min_distance = float('inf')   # 设定初始距离为无穷大
        
        # 找出当前data中的最小距离
        for i in range(len(data)):
            print("----")
            for j in range(i+1, len(data)):
                distance = euclidean_distance(data[i], data[j])
                print("计算 {} 与 {} 距离为 {}".format(data[i], data[j], distance))
                if distance < min_distance:
                    min_distance = distance
                    min_ij = (i, j)

        i, j = min_ij    # 最近数据点序号
        data1 = data[i]
        data2 = data[j]
        data = np.delete(data, j, 0)  # 删除原数据
        data = np.delete(data, i, 0)  # 删除原数据
        # b = np.atleast_2d([(data1[0]+data2[0])/2, (data1[1]+data2[1])/2])    # 计算两点新中心
        b = np.atleast_2d((data1+data2)/2)    # my 计算两点新中心 
        data = np.concatenate((data, b), axis=0)   # 将新数据点添加到data中
        print("最近距离: {} & {} = {}, 合并后中心: {}".format(data1, data2, min_distance, b))

    return data


if __name__ == '__main__':
    data = datasets.make_blobs(10, n_features=2, centers=2, random_state=10)
    # print(data[1])
    # plt.scatter(data[0][:,0], data[0][:, 1], c=data[1])
    # plt.show()

    res = agglomerative_clustering(data[0])
    print(res)

    ## sk ##
    # model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
    # print(model.fit_predict(data[0]))
