import numpy as np 
import pandas as pd 

def create_data():
    data_value = np.array([['long', 'thick', 175, 'no', 'man'], 
    					   ['short', 'medium', 168, 'no', 'man'],
	                       ['short', 'thin', 178, 'yes', 'man'], 
	                       ['short', 'thick', 172, 'no', 'man'],
	                       ['long', 'medium', 163, 'no', 'man'], 
	                       ['short', 'thick', 180, 'no', 'man'],
	                 	   ['long', 'thick', 173, 'yes', 'man'], 
	                 	   ['short', 'thin', 174, 'no', 'man'],
	                 	   ['long', 'thin', 164, 'yes', 'woman'], 
	                 	   ['long', 'medium', 158, 'yes', 'woman'],
	                 	   ['long', 'thick', 161, 'yes', 'woman'], 
	                 	   ['short', 'thin', 166, 'yes', 'woman'],
	                 	   ['long', 'thin', 158, 'no', 'woman'], 
	                 	   ['short', 'medium', 163, 'no', 'woman'],
	                 	   ['long', 'thick', 161, 'yes', 'woman'],
	                 	   ['long', 'thin', 164, 'no', 'woman'],
	                 	   ['short', 'medium', 172, 'yes','woman']])
    columns = np.array(['hair', 'voice', 'height', 'ear_stud','labels'])
    data=pd.DataFrame(data_value.reshape(17,5),columns=columns)
    return data

def create_data1():
	return pd.read_csv('5_1.csv')

def entropy(D):
	"""估计数据集(子集)的信息熵"""
	count = len(D)
	cate_count = {}
	for i in D:
		if i in cate_count:
			cate_count[i] += 1
		else:
			cate_count[i] = 1
	cate_count = np.array(list(cate_count.values()))
	p_i = cate_count / count
	
	return -np.sum(p_i * np.log2(p_i))

# def g(D, n):
	"""计算数据集D在第n个特征下的信息增益"""


if __name__ == '__main__':
	data = create_data1()
	# print(data)

	print(entropy(data['类别']))