"""
用梯度下降最小化一些简单函数
"""
def f1(x):
	"""
	待最小化的函数
	"""
	return x**2

def d1(x):
	"""
	导数
	"""
	return 2 * x

def train(x, f, d, learn_rate, epochs):
	"""
	Args:
		x: 自变量
		f: 函数
		d: 导数
		learn_rate: 学习率
		epochs: 迭代次数
	Returns:
	"""
	for epoch in range(epochs):
		x -= learn_rate * d(x)
	return x, f(x) 

if __name__ == '__main__':
	x = 10
	learn_rate = 0.01
	epochs = 1000
	print(train(x, f1, d1, learn_rate, epochs))