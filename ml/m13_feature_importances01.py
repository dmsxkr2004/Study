import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

import tensorflow as tf

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


model4 = DecisionTreeClassifier()
# model5 = RandomForestClassifier()


    
#3. 훈련
model4.fit(x_train, y_train)

#4. 평가, 예측

result = model4.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model4.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("RandomForestClassifier : ", result)
print("accuracy_score : ", acc)

print(model4.feature_importances_)
