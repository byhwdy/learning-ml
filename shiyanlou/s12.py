import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.metrics import accuracy_score

def perceptron_sgd(X, Y, alpha, epochs):
    """感知机随机梯度下降算法实现"""
    w = np.zeros(len(X[0]))   # 初始化参数为0
    b = np.zeros(1)

    for t in range(epochs):   # 迭代
        for i, x in enumerate(X):
            if ((np.dot(X[i], w)+b)*Y[i]) <= 0:    # 误分类点
                w = w + alpha*X[i]*Y[i]
                b = b + alpha*Y[i]

    return w, b


if __name__ == '__main__':
    df = pd.read_csv("course-12-data.csv", header=0)
    # print(df.head())

    plt.figure(figsize=(10, 6))
    plt.scatter(df['X0'], df['X1'], c=df['Y'])

    X = df[['X0', 'X1']].values
    Y = df['Y'].values
    alpha = 0.1
    epochs = 150
    w, b = perceptron_sgd(X, Y, alpha, epochs)
    # print(w, b)

    z = np.dot(X, np.array([w[0], w[1]]).T) + b
    y_pred = np.sign(z)
    # print(y_pred)

    # print(accuracy_score(Y, y_pred))

    x1_min, x1_max = df['X0'].min(), df['X0'].max(),
    x2_min, x2_max = df['X1'].min(), df['X1'].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = (np.dot(grid, np.array([w[0], w[1]]).T) + b).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0], linewidths=1, colors='red')
    plt.show()