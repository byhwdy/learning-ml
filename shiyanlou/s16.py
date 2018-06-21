from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt 
import numpy as np 

def random_k(k, data):
	"""初始化中心点
	参数:
	k: 中心点个数
	data: 数据集
	返回：
	init_centers: 初始化中心点
	"""
	# np.set_printoptions(precision=15)
	prng = np.random.RandomState(27)
	num_feature = np.shape(data)[1]
	init_centers = prng.randn(k, num_feature)*5  # 从方差为5的正态分布选取点
	# print(init_centers);exit()
	return init_centers

def d_euc(x, y):
	"""计算欧几里德距离
	""" 
	return np.sqrt(np.sum(np.power((x-y), 2)))

def update_center(clusters, data, centers):
	"""更新中心点
	参数: 
	clusters: 每一点分好的类别
	data： 数据集
	centers： 中心点集合
	返回：
	new_centers.reshape(num_centers, num_features): 新中心点集合
	"""
	num_centers = np.shape(centers)[0]  # 中心点个数
	num_features = np.shape(centers)[1]  # 特征数
	container = []
	for x in range(num_centers):
		each_container = []
		container.append(each_container)     # 容器，将相同类别数据存放在一起

	for i, cluster in enumerate(clusters):
		container[cluster].append(data[i])

	# 为方便计算，将list类型 转换为np.array类型
	container = np.array(list(map(lambda x: np.array(x), container)))

	new_centers = np.array([])  # 创建一个容器，存放中心点的坐标
	for i in range(len(container)):
		each_center = np.mean(container[i], axis=0) # 将每一个子集中数据均值作为中心点
		new_centers = np.append(new_centers, each_center)

	return new_centers.reshape(num_centers, num_features)  # 以矩阵方式返回中心点坐标

def kmeans_cluster(data, init_centers, k):
	"""K-Means
	参数：
	data：数据集
	init_centers： 初始化中心点集合
	k：中心点个数
	返回：
	centers_container： 每一次更新中心点集合
	cluster_container：每一次更新类别集合
	"""
	max_step = 50  # 定义最大迭代次数，中心点最多移动的次数
	# epsilon = 0.001 # 定义一个足够小的数， 通过中心点变化的距离是否小于该数，判断中心点是否变化
	epsilon = 0.3 # 定义一个足够小的数， 通过中心点变化的距离是否小于该数，判断中心点是否变化

	old_centers = init_centers

	centers_container = []  # 建立一个中心点容器，存放每一次变化后的中心点，以便后面绘图
	cluster_container = []  # 建立一个类别容器，存放每次中心点变化后的数据的类别
	centers_container.append(old_centers)

	for step in range(max_step):
		cluster = np.array([], dtype=int)
		for each_data in data:
			distances = np.array([])
			for each_center in old_centers:
				temp_distance = d_euc(each_data, each_center)  # 计算样本和中心点的欧氏距离
				distances = np.append(distances, temp_distance)
			lab = np.argmin(distances)  # 返回距离最近中心点的索引，即按照最近中心点分类
			cluster = np.append(cluster, lab)
		cluster_container.append(cluster)

		new_centers = update_center(cluster, data, old_centers)  # 根据子集分类更新中心点

		difference = np.fabs(new_centers-old_centers)
		
		print(new_centers, old_centers)
		print(difference)
		# print(difference.dtype)
		print(difference.any())
		# print(difference.any() < epsilon)
		# print(difference<epsilon)
		# if difference.any() < epsilon:  # 判断中心点是否移动
		if (difference < epsilon).all():  # 判断中心点是否移动
			return centers_container, cluster_container

		centers_container.append(new_centers)
		old_centers = new_centers

	return centers_container, cluster_container



















if __name__ == '__main__':
	# 数据
	blobs, _ = make_blobs(n_samples=200, centers=3, random_state=18)
	# print(blobs[:10])

	# 初始化3个中心点
	init_centers = random_k(3, blobs)
	# print(init_centers)

	# 计算最终中心点
	centers_container, cluster_container = kmeans_cluster(blobs, init_centers, 3)
	final_center = centers_container[-1]
	final_cluster = cluster_container[-1]
	# print(final_cluster.shape)
	
	# 画图
	plt.scatter(blobs[:, 0], blobs[:, 1], s=20, c=final_cluster)
	plt.scatter(final_center[:,0], final_center[:,1], s=100, marker='*', c='r')

	plt.show()
