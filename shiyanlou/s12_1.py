import numpy as np

def sigmoid(x):
	"""sigmoid函数"""
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
	"""sigmoid函数求导"""
	return sigmoid(x) * (1.0 - sigmoid(x))

if __name__ == '__main__':
	weights_demo = [np.ones([3,3]), np.ones([3, 1])]
	print(weights_demo)