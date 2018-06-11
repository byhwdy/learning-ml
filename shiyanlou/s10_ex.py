import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def sk_classify(x_train, y_train, x_test):
	model = MultinomialNB(alpha=1.0, fit_prior=True)
	model.fit(x_train, y_train)
	return model.predict(x_test)

def get_accuracy(y_test, y_pred):
	correct = np.sum(y_test == y_pred)
	n = len(y_test)
	return correct / n

# 原始数据
enterprise_data = pd.read_csv('course-10-company.csv')
enterprise_data = enterprise_data.replace({"P":1, "A":2, "N":3, "NB":0, "B":1})
# print(enterprise_data.head())

# 构建训练集，测试集
feature_data = enterprise_data.iloc[:, :-1]
label_data = enterprise_data['label']
x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.3, random_state=4)
# print(x_test)

y_pred = sk_classify(x_train, y_train, x_test)
# print(y_pred)
print(get_accuracy(y_test, y_pred))
