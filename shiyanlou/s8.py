from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd

def f1():
	"""用线性回归实现分类
	"""
	# 示例数据
	scores=[[1],[1],[2],[2],[3],[3],[3],[4],[4],[5],[6],[6],[7],[7],[8],[8],[8],[9],[9],[10]]
	passed= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]

	# 画图
	fig, axes = plt.subplots()
	axes.scatter(scores, passed, color='r')
	axes.set(xlabel='scores', ylabel='passed')

	model = LinearRegression()
	model.fit(scores, passed)

	x_temp = np.linspace(-2, 12, 100)
	axes.plot(x_temp, x_temp*model.coef_[0] + model.intercept_)
	plt.show() 

def plot_sigmoid():
	"""画出sigmoid函数的图像
	"""
	z = np.linspace(-12, 12, 100)
	sigmoid = 1 / (1 + np.exp(-z))
	fig, axes = plt.subplots()
	axes.plot(z, sigmoid)
	axes.set(xlabel='z', ylabel='sigmoid')
	plt.show()

#### 用python实现逻辑回归 ####
def sigmoid(z):
	"""模型
	"""
	return 1 / (1 + np.exp(-z))

def loss(h, y):
	"""对数损失函数
	"""
	return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

def gradient(X, h, y):
	"""对数损失函数的梯度
	"""
	return np.dot(X.T, h-y) / y.shape[0]

def Logistic_Regression(x, y, lr, num_iter):
	intercept = np.ones((x.shape[0], 1))
	x = np.concatenate((intercept, x), axis=1)   # 添加截距项
	w = np.zeros(x.shape[1])  # 初始化参数为0

	l = []  # 记录损失函数值

	for _ in range(num_iter):
		h = sigmoid(np.dot(x, w))   # 当前预测值

		g = gradient(x, h, y)   # 当前tidu

		w -= lr*g    # 对参数进行梯度下降

		# 当前损失函数
		current_loss = loss(sigmoid(np.dot(x, w)), y)
		l.append(current_loss)

	return l[len(l)-1], w, l


if __name__ == '__main__':
	# f1()
	# plot_sigmoid()

	# 数据
	df = pd.read_csv('course-8-data.csv', header=0)
	
	# 画出样本
	fig, axes = plt.subplots()
	axes.scatter(df['X0'], df['X1'], c=df['Y'])
	
	# 获得训练集
	x = df[['X0', 'X1']].values
	y = df['Y'].values

	## 测试python版logistic regression
	# # 训练参数
	# lr = 0.001
	# num_iter = 10000
	# L = Logistic_Regression(x, y, lr, num_iter)
	# # print(L)
	
	# # 画出决策边界
	# x1_min, x1_max = df['X0'].min(), df['X0'].max(),
	# x2_min, x2_max = df['X1'].min(), df['X1'].max(),
	# xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
	# grid = np.c_[xx1.ravel(), xx2.ravel()]
	# probs = (np.dot(grid, np.array([L[1][1:3]]).T) + L[1][0]).reshape(xx1.shape)
	# plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');

	# # 画出loss的下降过程
	# fig, axes = plt.subplots()
	# axes.plot(list(range(len(L[2]))), L[2])

	# plt.show()

	## sklearn实现逻辑回归
	model = LogisticRegression(tol=0.001, max_iter=10000)
	model.fit(x, y)
	# print(model.coef_, model.intercept_)
	
	# 画出决策边界
	# x1_min, x1_max = df['X0'].min(), df['X0'].max(),
	# x2_min, x2_max = df['X1'].min(), df['X1'].max(),
	# xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
	# grid = np.c_[xx1.ravel(), xx2.ravel()]
	# probs = (np.dot(grid, model.coef_.T) + model.intercept_).reshape(xx1.shape)
	# axes.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');

	# plt.show()

	# 在训练集上的分类准确率
	print(model.score(x, y))