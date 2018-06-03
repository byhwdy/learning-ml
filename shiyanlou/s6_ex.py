import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

## 原始数据
df = pd.read_csv('course-6-vaccine.csv', header=0)
# print(df)
# 绘图
# x = df['Year']
# y = df['Values']
# fig, axes = plt.subplots()
# axes.scatter(x, y)
# axes.plot(x, y, color='r')
# plt.show()

## 训练集，测试集
split_num = int(len(df)*0.7)
train_df = df[:split_num]
test_df = df[split_num:]
train_x = train_df['Year'].values 
train_y = train_df['Values'].values
test_x = test_df['Year'].values
test_y = test_df['Values'].values

## 用线性回归模型预测测试集
model = LinearRegression()
model.fit(train_x.reshape(len(train_x), 1), train_y.reshape(len(train_y), 1))
result = model.predict(test_x.reshape(len(test_x), 1))
# print(result)
# 评价
print("线性回归MAE:", mean_absolute_error(test_y.reshape(len(test_y), 1), result))
print("线性回归MSE:", mean_squared_error(test_y, result.flatten()))

## 用多项式模型
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_train_x = poly_features.fit_transform(train_x.reshape(len(train_x), 1))
poly_test_x = poly_features.fit_transform(test_x.reshape(len(test_x), 1))
model = LinearRegression()
model.fit(poly_train_x, train_y.reshape(len(train_y), 1))
result_poly = model.predict(poly_test_x)
# print(result_poly)
print("多项式回归MAE:", mean_absolute_error(test_y, result_poly.flatten()))
print("多项式回归MSE:", mean_squared_error(test_y, result_poly.flatten()))

# 再次处理数据
train_x = train_x.reshape(len(train_x),1)
test_x = test_x.reshape(len(test_x),1)
train_y = train_y.reshape(len(train_y),1)

## 管道
for m in [3, 4, 5]:
	model = make_pipeline(PolynomialFeatures(degree=m, include_bias=False), LinearRegression())
	model.fit(train_x, train_y)
	pred_y = model.predict(test_x)

	print("{}次多项式的MAE:".format(m), mean_absolute_error(test_y, pred_y.flatten()))
	print("{}次多项式的MSE:".format(m), mean_squared_error(test_y, pred_y.flatten()))
	print("=====")

## 阶数选择：计算 m 次多项式回归预测结果的 MSE 评价指标并绘图
mse = [] # 各阶数下的mse
m = 1 # 初始阶数
m_max = 10   # 最高阶数

while m <= m_max:
	model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
	model.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	mse.append(mean_squared_error(test_y, pred_y.flatten()))
	m += 1

print("MSE:", mse)

# 绘图
fig, axes = plt.subplots()
axes.scatter(list(range(1, m_max+1)), mse)
axes.plot(list(range(1, m_max+1)), mse, 'r')

plt.show()
