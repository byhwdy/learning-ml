import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import numpy as np

def choice_2(x):
	x = int(x)
	if x < 5:
		return 'bad'
	elif x >= 5 and x < 10:
		return 'medium'
	elif x >= 10 and x < 15:
		return 'good'
	else:
		return 'excellent'

def choice_3(x):
    x=int(x)
    if x>3:
        return "high"
    elif x>1.5:
        return "medium"
    else:
        return "low"

def replace_feature(data):
	"""特征值适配sklearn"""
	for each in data.columns:
		feature_list = data[each]
		unique_value = set(feature_list)
		i = 0
		for fea_value in unique_value:
			data[each] = data[each].replace(fea_value, i)
			i += 1
	return data



if __name__ == '__main__':
	stu_grade = pd.read_csv('course-13-student.csv')
	# print(stu_grade.head())
	new_data=stu_grade.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,14,15,24,25,26]]
	# print(new_data.head())
	stu_data = new_data.copy()
	stu_data['G1'] = pd.Series(map(lambda x: choice_2(x), stu_data['G1']))
	stu_data['G2'] = pd.Series(map(lambda x: choice_2(x), stu_data['G2']))
	stu_data['G3'] = pd.Series(map(lambda x: choice_2(x), stu_data['G3']))
	# print(stu_data.head())
	stu_data["Pedu"]=pd.Series(map(lambda x:choice_3(x),stu_data["Pedu"]))
	# print(stu_data.head())
	stu_data = replace_feature(stu_data)
	# print(stu_data.head(10))

	x_train, x_test, y_train, y_test = train_test_split(stu_data.iloc[:, :-1], stu_data['G3'], test_size=0.3, random_state=5)
	# print(x_test)

	dt_model = DecisionTreeClassifier(criterion='entropy', random_state=34)
	dt_model.fit(x_train, y_train)

	# 画图
	img = export_graphviz(
	    dt_model, out_file=None,    
	    feature_names=stu_data.columns[:-1].values.tolist(),  #传入特征名称
	    class_names=np.array(["bad","medium","good","excellent"]), #传入类别值
	    filled=True, node_ids=True,
	    rounded=True)

	dot = graphviz.Source(img) #展示决策树
	dot.render('zsh.gv', view=True)