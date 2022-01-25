import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [np.nan, 4, np.nan, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]])


# print(data.shape)  # (4, 5)

data = data.transpose()  # (4, 5) => (5, 4)
data.columns = ['a','b','c','d']  # column명 지정하기


# SimpleImputer는 총 4가지의 파라미터가 있다.

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer  

# imputer = SimpleImputer(strategy='mean')   # mean으로 결측치를 채움
# imputer = SimpleImputer(strategy='median')   # 중위값으로 결측치를 채움
# imputer = SimpleImputer(strategy='most_frequent')   # 제일 많이 사용한 값으로 결측치를 채움
# imputer = SimpleImputer(strategy='constant')  # 0으로 결측치를 채움
imputer = SimpleImputer(strategy='constant', fill_value=777)  # 0으로 채워진 결측치를 다른 숫자로 채우고 싶을 때 쓰는 파라미터

imputer.fit(data)
data2 = imputer.transform(data)
print(data2)

################################################## 특정 칼럼 결측치 처리 #########################################################

means = data['a'].mean()
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()
print(meds)
data['b'] = data['b'].fillna(meds)
print(data)

imputer.fit(data)
data3 = imputer.transform(data)

print(data3)
'''
[[  2.           2.         777.         777.        ]
 [  6.66666667   4.           4.           4.        ]
 [  6.66666667   4.         777.         777.        ]
 [  8.           8.           8.           8.        ]
 [ 10.           4.          10.         777.        ]]
'''