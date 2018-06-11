import pandas as pd 

df = pd.read_csv('challenge-6-abalone.csv')
# print(df.tail())
tmp_df = pd.DataFrame([df.columns.values])
df = pd.concat([tmp_df, df[:-1]], ignore_index=True)
print(pd.concat([df[:2], df[-2:]]))