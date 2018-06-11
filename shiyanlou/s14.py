import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def get_accuracy(test_labels, pred_labels):
	"""准确率"""
	correct = np.sum(test_labels==pred_labels)
	n = len(test_labels)
	return correct / n 


if __name__ == '__main__':
	stu_data = pd.read_csv('course-14-student.csv')
	# print(stu_data.head())

	x_train, x_test, y_train, y_test = train_test_split(stu_data.iloc[:, :-1], stu_data['G3'], test_size=0.3, random_state=35)
	# print(x_test.head(10))

	dt_model = DecisionTreeClassifier(criterion='entropy', random_state=34)
	dt_model.fit(x_train, y_train)
	dt_y_pred = dt_model.predict(x_test)
	# print(dt_y_pred)
	# print(get_accuracy(y_test, dt_y_pred))

	tree = DecisionTreeClassifier(criterion='entropy', random_state=28)
	bag = BaggingClassifier(tree, n_estimators=100, max_samples=1.0, random_state=3)
	bag.fit(x_train, y_train)
	bt_y_pred = bag.predict(x_test)
	# print(bt_y_pred)
	# print(get_accuracy(y_test, bt_y_pred))

	rf = RandomForestClassifier(n_estimators=100, max_features=None, criterion='entropy')
	rf.fit(x_train, y_train)
	rf_y_pred = rf.predict(x_test)
	# print(rf_y_pred)
	# print(get_accuracy(y_test, rf_y_pred))

	ad = AdaBoostClassifier(n_estimators=100)
	ad.fit(x_train, y_train)
	ad_y_pred = ad.predict(x_test)
	# print(ad_y_pred)
	# print(get_accuracy(y_test, ad_y_pred))

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=33)
	clf.fit(x_train, y_train)
	gt_y_pred = clf.predict(x_test)
	print(gt_y_pred)
	print(get_accuracy(y_test, gt_y_pred))
