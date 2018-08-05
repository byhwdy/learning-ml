import numpy as np 
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing() 
m, n = housing.data.shape
housing_feature = np.append(housing.data, np.ones((m, 1)) , axis=1)

print(housing_feature)

X = tf.constant(housing_feature, dtype=tf.float32, name='X')

# Y = 