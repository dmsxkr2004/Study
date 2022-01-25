import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.pipeline import make_pipeline, Pipeline
# model7 = SVC()
model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = model.score(x_test, y_test)  

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("RandomForestRegressor : ", result)
print("r2_score : ", r2)

'''
RandomForestRegressor :  0.3896812551799238
r2_score :  0.3896812551799238
'''