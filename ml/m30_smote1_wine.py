import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

print(pd.Series(y).value_counts()) # 개별 판다스로  y 값 찍어주는법 np.unique(y, return_counts=True)
# 1    71
# 0    59
# 2    48

print(y)

# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
x_new = x[:-30]
y_new = y[:-30]
# 1    71
# 0    59
# 2    18
print(pd.Series(y_new).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, random_state=66, shuffle = True, train_size = 0.75, stratify = y_new)
print(pd.Series(y_train).value_counts())
'''
1    53
0    44
2    14
dtype: int64
'''
#2. 모델구성
model = XGBClassifier(n_jobs = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
score = model.score(x_test, y_test)
print("model.score : ",round(score,4))
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score", round(acc,4))
# accuracy_score 0.9778 -> 그냥 돌린 값

# accuracy_score 0.9459 -> 데이터 빼고 돌린 값
print("==============SMOTE 적용===================")
smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train) # 훈련하는 수치라서 핏해줌 , 테스트 데이터는 비교데이터이기때문에 건들면안됨

#2. 모델구성
model = XGBClassifier(n_jobs = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
score = model.score(x_test, y_test)
print("model.score : ",round(score,4))
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score", round(acc,4))

