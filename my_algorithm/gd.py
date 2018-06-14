import numpy as np  
import matplotlib.pyplot as plt 

# x^2函数
def f(x):
	return np.power(x, 2)

# x^2函数的梯度1
def d_f_1(x):
	return 2.0 * x

# x^2函数的梯度2
def d_f_2(f, x, delta=1e-4):
	return (f(x+delta) - f(x-delta)) / (2 * delta)

if __name__ == '__main__':
	# 示例数据
	# xs = np.arange(-10, 11)

	# 画图
	# plt.plot(xs, f(xs))
	# plt.show()
	
	# 梯度下降求最小值
	lr = 0.1
	epochs = 30
	x_init = 10.0

	x_tmp = x_init
	for i in range(epochs):
		# x_tmp = x_tmp - lr * d_f_1(x_tmp)
		x_tmp = x_tmp - lr * d_f_2(f, x_tmp)
		print('====={}====='.format(i))
		print('x: ', x_tmp)
		print('f(x): ', f(x_tmp))

	print('x_init = ', x_init)
	print('arg min f(x) of x =', x_tmp)
	print('min f(x) = ', f(x_tmp))
