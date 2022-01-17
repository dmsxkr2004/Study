import numpy as np
import pandas as pd
from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
#1. 데이터

datasets = load_diabetes()
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
model7.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = model7.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model7.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("SVC : ", result)
print("r2_score : ", r2)
'''
Perceptron :  0.0
r2_score :  -1.7052375708577001
'''
'''
KNeighborsRegressor :  0.3968391279034368
r2_score :  0.3968391279034368
'''
'''
LinearRegression :  0.5063891053505035
r2_score :  0.5063891053505035
'''
'''
DecisionTreeRegressor :  -0.1910851914726246
r2_score :  -0.1910851914726246
'''
'''
RandomForestRegressor :  0.3828175703033072
r2_score :  0.3828175703033072
'''
'''
LinearSVC :  0.0
r2_score :  -0.05417705666534589
'''
'''
SVC :  0.0
r2_score :  -0.10922943120678053
'''

