from scipy.linalg import hilbert
import numpy as np 
import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import Ridge, Lasso
from matplotlib import pyplot as plt

## 构造一个具有多重共线性的数据集
# 获取希尔伯特矩阵
x = hilbert(10)
# print(x)
# 查看其转置矩阵与原矩阵的乘积
# print(np.matrix(x).T * np.matrix(x))   ## 约为0
# 计算皮尔逊相关系数
# print(pd.DataFrame(x, columns=['x%d' % i for i in range(1, 11)]).corr())   ## 接近1

# 假设希尔伯特矩阵构成的数据集服从线性分布，获取y值
np.random.seed(10)
w = np.random.randint(2, 10, 10)
y_temp = np.matrix(x) * np.matrix(w).T
y = np.array(y_temp.T)[0]
print("实际参数 w: ", w)
print("实际函数值 y: ", y)

def f1():
	"""用希尔伯特矩阵列向量之间的多重共线性演示最小二乘法的局限性
	"""

	# 最小二乘法求解
	func = lambda p, x: x.dot(p)   # 函数公式
	err_func = lambda p, x, y: func(p, x) - y # 残差公式 
	p_init = np.random.randint(1, 2, 10)   # 初始化p全为1
	parameters = leastsq(err_func, p_init, args=(x, y))   # 用scipy的leastsq求解
	print("拟合参数 w: ", parameters[0])

def f2():
	"""岭回归
	"""
	model = Ridge(fit_intercept=False)
	model.fit(x, y)
	print(model.coef_)

def f3():
	"""岭回归lamda参数选择
	"""
	alphas = np.linspace(-3, 2, 20)

	coefs = []
	for a in alphas:
		ridge = Ridge(alpha=a, fit_intercept=False)
		ridge.fit(x, y)
		coefs.append(ridge.coef_)
	# print(alphas.shape, np.array(coefs).shape)
	fig, axes = plt.subplots()
	axes.plot(alphas, coefs)
	# axes.scatter(np.linspace(0, 0, 10), f1.parameters[0])
	plt.show()

def f4():
	"""lasso
	"""
	alphas = np.linspace(-2, 2, 10)
	
	lasso_coefs = []
	for a in alphas:
		lasso = Lasso(a, fit_intercept=False)
		lasso.fit(x, y)
		lasso_coefs.append(lasso.coef_)

	fig, axes = plt.subplots()
	axes.plot(alphas, lasso_coefs)
	plt.show()


if __name__ == '__main__':
	# f1()
	# f2()
	# f3()
	f4()