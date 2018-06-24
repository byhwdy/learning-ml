import numpy as np  
from sklearn import datasets
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans, Birch, DBSCAN 


def euclidean_distance(a, b):
	return np.sqrt(np.sum(np.square(a-b)))

def search_neighbors(D, P, eps):
	"""找出数据集中点P的eps邻域
	"""
	neighbors = []
	for Pn in range(len(D)):
		if euclidean_distance(D[Pn], D[P]) < eps:
			neighbors.append(Pn)
	return  neighbors

def dbscan_cluster(D, eps, MinPts):
	labels = [0]*len(D) # 初始化数据集中的数据类别全部为0
	C = 0
	for P in range(0, len(D)):
		# 当前点已分类，不处理
		if not (labels[P] == 0):
			continue

		## 当前点未分类
		# 确定其eps邻域
		Neighbors = search_neighbors(D, P, eps)

		# 非核心点标记为-1
		if len(Neighbors) < MinPts:
			labels[P] = -1
		# 核心点作为新类别中心
		else:
			C += 1   # 原类别点+1作为新类别的标签
			labels[P] = C # 给当前的核心点设置新的类别

			# 
			for i, n in enumerate(Neighbors):
				Pn = Neighbors[i]   # P邻域中的当前点

				# 设置P邻域中的点的类别
				if labels[Pn] == 0:
					labels[Pn] = C

					# 进一步搜索P的邻居的邻居
					PnNeighbors = search_neighbors(D, Pn, eps)
					if len(PnNeighbors) >= MinPts:  # 如果满足密度阈值要求则联通
						Neighbors += PnNeighbors
				elif labels[Pn] == -1:  # 如果该点曾被标记为-1， 则重新连接到类别中
					labels[Pn] = C
	return labels				

if __name__ == '__main__':
	noisy_moons, _ = datasets.make_moons(n_samples=100, noise=.05, random_state=10)
	# print(noisy_moons[:5], noisy_moons.shape)

	# plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1])
	# plt.show()

	# kmeans_c = KMeans(n_clusters=2).fit_predict(noisy_moons)
	# birch_c = Birch(n_clusters=2).fit_predict(noisy_moons)
	# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
	# axes[0].scatter(noisy_moons[:, 0], noisy_moons[:, 1], c=kmeans_c, cmap='bwr')
	# axes[1].scatter(noisy_moons[:, 0], noisy_moons[:, 1], c=birch_c, cmap='bwr')
	# axes[0].set_xlabel('K-Means')
	# axes[1].set_xlabel('BIRCH')
	# plt.show()

	# dbscan_c = dbscan_cluster(noisy_moons, eps=0.5, MinPts=5)
	# print(np.array(dbscan_c))
	# plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1], c=dbscan_c, cmap='bwr')
	# plt.show()

	## sk ##
	dbscan_sk = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
	dbscan_sk_c = dbscan_sk.fit_predict(noisy_moons)
	print(dbscan_sk_c)
	plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1], c=dbscan_sk_c, cmap='bwr')
	plt.show()