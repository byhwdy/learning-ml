import numpy as np

def sigmoid(x):
	"""sigmoid函数"""
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
	"""sigmoid函数求导"""
	return sigmoid(x) * (1.0 - sigmoid(x))

if __name__ == '__main__':
	# 初始化权重
	weights_demo = [np.ones([3,3]), np.ones([3, 1])]
	# print(weights_demo)

	# 定义一组示例数据
	X = np.array([2,3])

	# 添加截距项
	X = np.array([1,2,3])

	a = [X]   # 预留给下一步操作

	# 前向传播
	for l in range(len(weights_demo)):
		z = np.dot(a[l], weights_demo[l])
		print("第{}层线性组合计算结果：\n".format(l), z)

		activation = sigmoid(z)
		print("第{}层激活函数计算结果：\n".format(l), activation)

		a.append(activation)
