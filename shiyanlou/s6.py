from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

## 数据
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]

## 作图
fig, axes = plt.subplots()
axes.scatter(x, y)


#### 三次多项式
## 三次多项式模型及残差函数
def func(p, x):
    w0, w1, w2 = p
    return w0 + w1*x + w2*x*x

def err_func(p, x, y):
    return func(p, x) - y

p_init = np.random.randn(3)
parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
# 参数
print("三次多项式参数：", parameters[0])
# 画出拟合曲线
x_temp = np.linspace(0, 80, 10000)
axes.plot(x_temp, func(parameters[0], x_temp), 'r')


#### n次多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

def err_func(p, x, y):
    return fit_func(p, x) - y

def n_poly(n):
    p_init = np.random.randn(n)
    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
    return parameters[0]
# 参数
print("n=3阶多项式参数：", n_poly(3))

# 作图
x_temp = np.linspace(0, 80, 10000)
fig, axes = plt.subplots(2, 3, figsize=(15,10))

axes[0,0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')
axes[0,0].scatter(x, y)
axes[0,0].set_title("m = 4")

axes[0,1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')
axes[0,1].scatter(x, y)
axes[0,1].set_title("m = 5")

axes[0,2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')
axes[0,2].scatter(x, y)
axes[0,2].set_title("m = 6")

axes[1,0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')
axes[1,0].scatter(x, y)
axes[1,0].set_title("m = 7")

axes[1,1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')
axes[1,1].scatter(x, y)
axes[1,1].set_title("m = 8")

axes[1,2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')
axes[1,2].scatter(x, y)
axes[1,2].set_title("m = 9")


#### sklearn多项式拟合
x = np.array(x).reshape(len(x), 1)
y = np.array(y).reshape(len(y), 1)
poly_x = PolynomialFeatures(degree=5, include_bias=False).fit_transform(x)
# print(poly_x)

model = LinearRegression()
model.fit(poly_x, y)

x_temp = np.array(x_temp).reshape(len(x_temp), 1)
poly_x_temp = PolynomialFeatures(degree=5, include_bias=False).fit_transform(x_temp)
fig, axes = plt.subplots()
axes.scatter(x, y)
axes.plot(x_temp, model.predict(poly_x_temp), 'r')

plt.show()