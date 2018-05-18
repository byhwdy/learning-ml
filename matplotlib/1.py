from matplotlib import pyplot as plt
import numpy as np 

# 数据
x = np.linspace(0, 10, 20)

# fig,axes对象
# fig, axes = plt.subplots()
# fig, axes = plt.subplots(figsize=(12,6))
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,12))

# x,y轴，标题
# axes.set_xlabel('X')
# axes.set_ylabel('Y')
# axes.set_title('TITLE')

## 线形图，plot
# 颜色，透明度 
# axes.plot(x, x+1, 'red', alpha=0.4)
# axes.plot(x, x+2, color='b')
# axes.plot(x, x+3, color='#15cc55')

# 线宽
# axes.plot(x, x+1, color='blue', linewidth=0.25)
# axes.plot(x, x+2, color='blue', linewidth=0.50)
# axes.plot(x, x+3, color='blue', linewidth=1.00)
# axes.plot(x, x+4, color='blue', linewidth=2.00)

# 虚线线型
# axes.plot(x, x+1, color='blue', lw=2.00, linestyle='-')
# axes.plot(x, x+2, color='blue', lw=2.00, ls='-.')
# axes.plot(x, x+3, color='blue', lw=2.00, ls=':')

# 虚线交错宽度
# line, = axes.plot(x, x, color='black', lw=1.50)
# line.set_dashes([5, 10, 15, 10])

# 符号
# axes.plot(x, x+1, color='green', lw=2, ls='--', marker='+')
# axes.plot(x, x+2, color='green', lw=2, ls='--', marker='o')
# axes.plot(x, x+3, color='green', lw=2, ls='-', marker='s')
# axes.plot(x, x+4, color='green', lw=2, ls=':', marker='1')

# 符号大小和颜色
# axes.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
# axes.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
# axes.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
# axes.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 
#         markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")

# 显示网格
# axes[0].plot(x, x**2, x, x**3, lw=2)
# axes[0].grid(True)

# 设置坐标轴范围
# axes[1].plot(x, x**2, x, x**3)
# axes[1].set_ylim([0, 60])
# axes[1].set_xlim([2, 5])



# 图例
# axes.legend(['y = x**2', 'y = x**3'], loc=2)

# n = np.array([0,1,2,3,4,5])

# fig, axes = plt.subplots(1, 4, figsize=(16, 5))

# # 散点图
# axes[0].scatter(x, x+0.25*np.random.randn(len(x)))
# axes[0].set_title('scatter')

# # 梯步图
# axes[1].step(n, n**2, lw=2)
# axes[1].set_title('step')

# # 条形图
# axes[2].bar(n, n**2, align='center', width=0.5, alpha=0.4)
# axes[2].set_title('bar')

# # 面积图
# axes[3].fill_between(x, x**2, x**3, color='green', alpha=0.5)
# axes[3].set_title('fill_between')

## 雷达图
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_axes([0.1, 0.1, .8, .8], polar=True)
# t = np.linspace(0, 2*np.pi, 100)
# ax.plot(t, t, color='g', lw=2)

## 直方图
# n = np.random.randn(100000)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# axes[0].hist(n)
# axes[0].set_title('Default histogram')
# axes[0].set_xlim((min(n), max(n)))

# axes[1].hist(n, bins=100)
# # axes[1].hist(n, cumulative=True, bins=100)
# axes[1].set_title('cumulative detailed histogram')
# axes[1].set_xlim(min(n), max(n))

## 等高线图
alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
	return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)

phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

# fig, ax = plt.subplots()
# cnt = ax.contour(Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0,1,0,1])

## 3D图形
from mpl_toolkits.mplot3d.axes3d import Axes3D 

# fig = plt.figure(figsize=(14, 6))
# ax = fig.add_subplot(1,2,1, projection='3d')
# ax.plot_surface(X,Y,Z, rstride=4, cstride=4, linewidth=0)

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=plt.cm.coolwarm)

ax.set_xlim3d(-np.pi, 2*np.pi)
ax.set_ylim3d(0, 3*np.pi)
ax.set_zlim3d(-np.pi, 2*np.pi)


plt.show()