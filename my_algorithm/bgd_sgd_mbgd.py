"""
以线性回归为例试验bgd，sgd，mbgd
"""
import numpy as np 
import random

def batch_gradient_descent(x, y, theta, m, alpha, max_iters):
	"""批量梯度下降
	x: 训练集
	y：lables
	theta：参数
	m：样本数
	alpha：学习率
	max_iters：迭代次数
	"""
	for i in range(max_iters):
		grad = np.dot(x.T, np.dot(x, theta) - y) / m 
		theta -= alpha * grad

	return theta

def stochastic_gradient_descent(x, y, theta, m, alpha, max_iters):
	"""随机梯度下降
	"""
	for i in range(max_iters):
		rand_index = random.sample(range(m), 1)[0]
		choice_x = x[rand_index]
		choice_y = y[rand_index]
		grad = (np.dot(choice_x, theta) - choice_y) * choice_x
		theta -= alpha * grad

	return theta

def predict(x, theta):
	return np.dot(x, theta)

if __name__ == '__main__':
	train_x = np.array([[1., 1.1, 1.5], [1., 1.3, 1.9], [1., 1.5, 2.3], [1., 1.7, 2.7], [1., 1.9, 3.1], [1., 2.1, 3.5], [1., 2.3, 3.9], [1., 2.5, 4.3], [1., 2.7, 4.7], [1., 2.9, 5.1]])

	train_y = np.array([2.5, 3.2, 3.9, 4.6, 5.3, 6., 6.7, 7.4, 8.1, 8.8])
	m, n = train_x.shape

	# 初始化theta
	theta = np.ones(n)
	# theta = np.random.random(n)

	# 超参数
	alpha = 0.1
	max_iters = 5000

	# 训练：批量梯度下降求参数
	theta1 = batch_gradient_descent(train_x, train_y, theta, m, alpha, max_iters)
	print('batch_gd: theta=', theta1)

	# 训练：随机梯度下降求参数
	theta2 = stochastic_gradient_descent(train_x, train_y, theta, m, alpha, max_iters)
	print('stochastic_gd: theta=', theta2)

	# 评价
	x = [[1., 3.1, 5.5], [1., 3.3, 5.9], [1., 3.5, 6.3], [1., 3.7, 6.7], [1., 3.9, 7.1]]
	print(predict(x, theta1))
	print(predict(x, theta2))
