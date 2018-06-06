import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from time import time

def sklearn_knn(x_train, y_train, x_test, k):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train, y_train)
	pred_y = knn.predict(x_test)
	return pred_y

def get_accuracy(test_label, pred_label):
    """计算精确度
    """
    return np.sum(test_label == pred_label) / len(test_label)

if __name__ == '__main__':
	## 数据
	data = pd.read_csv('course-9-syringa.csv')

	## 画图分析数据
	# （略）

	## 构建训练集，测试集
	features = data.iloc[:, :-1]
	targets = data['labels']
	x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=2)
	# print(x_test)

	## knn预测
	# y_pred = sklearn_knn(x_train, y_train, x_test, 3)
	# print(y_pred)

	## 计算精确度
	# print(get_accuracy(y_test, y_pred))
	
	## k值选择
	# normal_accuracy = []
	# k_values = range(2, 11)   # 2-10
	# for k in k_values:
	# 	y_pred = sklearn_knn(x_train, y_train, x_test, k)
	# 	normal_accuracy.append(get_accuracy(y_test, y_pred))

	# plt.xlabel('k')
	# plt.ylabel('accuracy')
	# new_ticks = np.linspace(0.6, 0.9, 10)
	# plt.yticks(new_ticks)
	# plt.plot(k_values, normal_accuracy, c='r')
	# plt.grid(True)
	# plt.show()
	
	## 使用kd树
	start1 = time()
	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	end1 = time()
	print('normal time: ', end1 - start1)

	start2 = time()
	kd_knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
	kd_knn.fit(x_train, y_train)
	y_pred = kd_knn.predict(x_test)
	end2 = time()
	print('kd time: ', end2 - start2)