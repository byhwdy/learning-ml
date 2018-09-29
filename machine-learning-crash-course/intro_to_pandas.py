from __future__ import print_function
import pandas as pd 

# print(pd.__version__)

# pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

# city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
# population = pd.Series([852469, 1015785, 485199])
# pd.DataFrame({ 'City name': city_names, 'Population': population })

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
t = california_housing_dataframe.describe()
t = california_housing_dataframe.head()
california_housing_dataframe.hist('housing_median_age')
# print(t)