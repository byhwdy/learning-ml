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
	prng = np.random.RandomState(27)
	num_feature = np.shape(data)[1]
	init_centers = prng.randn(k, num_feature)*5  # 从方差为5的正态分布选取点
	return init_centers

def d_euc(x, y):
	"""计算欧几里德距离
	""" 
	return np.sqrt(np.sum(np.power((x-y), 2)))
























if __name__ == '__main__':
	# 数据
	blobs, _ = make_blobs(n_samples=200, centers=3, random_state=18)
	# print(blobs[:10])

	# 初始化3个中心点
	init_centers = random_k(3, blobs)
	# print(init_centers)
	

	# 画图
	plt.scatter(blobs[:, 0], blobs[:, 1], s=20)
	plt.scatter(init_centers[:,0], init_centers[:,1], s=100, marker='*', c='r')
	plt.show()