import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

print(data.shape) # (4, 5)
data = data.transpose()
data.columns = ['a', 'b', 'c', 'd']
print(data)

# print(data.isnull()) # True 지점이 결측치이다
# print(data.isnull().sum())
# print(data.info())

# #1. 결측치 삭제

# print(data.dropna())
# print(data.dropna(axis = 0))
# print(data.dropna(axis = 1))

#2-1. 특정값 - 평균값

means = data.mean()
print(means) 
data2 = data.fillna(means)
print(data)

#2-2. 특정값 - 중위값

meds = data.median()
print(meds)
data2 = data.fillna(meds)
print(data)

#2-3. 특정값 - ffill, bfill

data2 = data.fillna(method = 'ffill')
print(data2)
data2 = data.fillna(method = 'bfill')
print(data2)

data2 = data.fillna(method = 'ffill', limit = 1)
print(data2)
data2 = data.fillna(method = 'bfill', limit = 1)
print(data2)

#2-4. 특정값 - 채우기

data2 = data.fillna(747474)
print(data2)

###################################### 특정 컬럼만 !!##################################################

means = data['a'].mean()
print(means)

data['a'] = data['a'].fillna(means)
print(data)

meds = data['a'].median()
print(meds)

data['b'] = data['b'].fillna(meds)
print(data)