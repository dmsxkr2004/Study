import numpy as np
import pandas as pd
from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
#1. 데이터

datasets = load_boston()
# print(datasets.DESCR)      # x = (150,4) y = (150,1)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (150,4) (150,)
print(y)
# print(np.unique(y))     #   [0, 1, 2]

import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model1 = Perceptron()
model2 = KNeighborsRegressor()
model3 = LinearRegression()
model4 = DecisionTreeRegressor()
model5 = RandomForestRegressor()
model6 = LinearSVC()
model7 = SVC()

#3. 훈련
model6.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = model6.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model6.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("LinearSVC : ", result)
print("r2_score : ", r2)
'''
Perceptron :  0.6388888888888888
accuracy_score :  0.6388888888888888
'''
'''
KNeighborsRegressor :  0.5900872726222293
r2_score :  0.5900872726222293
'''
'''
LinearRegression :  0.8111288663608656
r2_score :  0.8111288663608656
'''
'''
DecisionTreeRegressor :  0.8109908352017086
r2_score :  0.8109908352017086
'''
'''
RandomForestRegressor :  0.9219446869013606
r2_score :  0.9219446869013606
'''
'''
LinearSVC :  0.5277777777777778
accuracy_score :  0.5277777777777778
'''
'''
SVC :  0.6944444444444444
accuracy_score :  0.6944444444444444
'''

