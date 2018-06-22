from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt 
from s16 import *


def get_sum_dis(centers, data):
	"""样本与最近中心点距离之和
	参数：
	centers：中心点集合
	data：数据集
	返回：
	np.sum(dis_container): 样本距离最近中心点距离之和
	dis_container: 样本距离最近中心点的距离集合
	"""
	dis_container = np.array([])
	for each_data in data:
		min_distance = 0 
		for each_center in centers:
			if min_distance == 0 or min_distance > d_euc(each_data, each_center):
				min_distance = d_euc(each_data, each_center)
		dis_container = np.append(dis_container, min_distance)
	return np.sum(dis_container), dis_container

def get_init_center(data, k):
	"""k_means++ 初始化中心点的方法
	"""		
	seed = np.random.RandomState(20)
	first_center = data[seed.randint(0, len(data))]

	centers_container = [first_center]

	for i in range(k-1):
		sum_dis, dis_con = get_sum_dis(centers_container, data)
		r = np.random.randint(0, sum_dis)
		for j in range(len(dis_con)):
			r = r - dis_con[j]
			if r <= 0:
				centers_container.append(data[j])
				break

	return np.array(centers_container)





























 
if __name__ == '__main__':
	# 生成数据
	blobs_plus, _ = make_blobs(n_samples=800, centers=5, random_state=18)
	
	## k_means ##
	km_init_center = random_k(5, blobs_plus)
	km_centers, km_clusters = kmeans_cluster(blobs_plus, km_init_center, 5)
	km_final_center = km_centers[-1]
	km_final_cluster = km_clusters[-1]
	# plt.scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c=_)
	# plt.scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c=km_final_cluster)  # 可视化数据
	# plt.scatter(km_init_center[:, 0], km_init_center[:, 1], s=100, marker='*', c='r')
	# plt.scatter(km_final_center[:, 0], km_final_center[:, 1], s=100, marker='o', c='r')	
	# plt.show()
	
	## k_means++ ##
	plus_init_center = get_init_center(blobs_plus, 5)
	# print(plus_init_center)
	## k_means++ 选取中心点的过程 ##
	# num = len(plus_init_center)

	# fig, axes = plt.subplots(1, num, figsize=(25, 4))

	# axes[0].scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c='b')
	# axes[0].scatter(plus_init_center[0, 0], plus_init_center[0, 1], s=100, marker='*', c='r')
	# axes[0].set_title('first_center')

	# for i in range(1, num):
	# 	axes[i].scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c='b')
	# 	axes[i].scatter(plus_init_center[:i+1, 0], plus_init_center[:i+1, 1], s=100, marker='*', c='r')
	# 	axes[i].set_title('step{}'.format(i))

	# axes[-1].scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c='b')
	# axes[-1].scatter(plus_init_center[:, 0], plus_init_center[:, 1], s=100, marker='*', c='r')
	# axes[-1].set_title('final_center')

	# plt.show()
	#### 
	plus_centers, plus_clusters = kmeans_cluster(blobs_plus, plus_init_center, 5)
	plus_final_center = plus_centers[-1]
	plus_final_cluster = plus_clusters[-1]
	plt.scatter(blobs_plus[:, 0], blobs_plus[:, 1], s=20, c=plus_final_cluster)
	plt.scatter(plus_final_center[:, 0], plus_final_center[:, 1], s=100, marker='*', c="r")
	plt.show()








	