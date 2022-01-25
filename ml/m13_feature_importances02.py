#13_1 번을 카피
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target
x = pd.DataFrame(x)
x = x.drop([0], axis=1)
x.info()
x.to_numpy()
print(type(x))
import tensorflow as tf

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier


model4 = DecisionTreeClassifier(max_depth = 3, random_state=66)
# model5 = RandomForestClassifier(max_depth = 3, random_state=66)
# model6 = XGBClassifier()
# model7 = GradientBoostingClassifier()



#3. 훈련
model4.fit(x_train, y_train)

#4. 평가, 예측

result = model4.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model4.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("DecisionTreeClassifier : ", result)
print("accuracy_score : ", acc)

print(model4.feature_importances_)
'''
DecisionTreeClassifier :  0.9
accuracy_score :  0.9
[0.         0.57385373 0.42614627]
'''
'''
RandomForestClassifier :  1.0
accuracy_score :  1.0
[0.14884697 0.40483696 0.44631606]
'''
'''
XGBClassifier :  0.9
accuracy_score :  0.9
[0.02876592 0.6337989  0.33743513]
'''
'''
GradientBoostingClassifier :  0.9333333333333333
accuracy_score :  0.9333333333333333
[0.01645295 0.29869528 0.68485177]
'''