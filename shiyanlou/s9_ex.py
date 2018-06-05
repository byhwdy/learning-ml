import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 数据
data = pd.read_csv('course-9-syringa.csv')

# 画图分析数据
# （略）

# 构建训练集，测试集
features = data.iloc[:, :-1]
targets = data['labels']
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=2)
# print(x_test)

def sklearn_knn(x_train, y_train, x_test, k):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train, y_train)
	pred_y = knn.predict(x_test)
	return pred_y

y_pred = sklearn_knn(x_train, y_train, x_test, 3)
print(y_pred)