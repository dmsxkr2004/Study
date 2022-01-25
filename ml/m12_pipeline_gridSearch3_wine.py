import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# parameters = [
#     {'randomforestclassifier__max_depth' : [6, 8, 10],
#      'randomforestclassifier__min_samples_leaf' : [3, 5, 7]},
#     {'randomforestclassifier__min_samples_leaf' : [3, 5, 7],
#     'randomforestclassifier__max_samples_split': [3, 5, 10]},
# ]
parameters = [
    {'rf__max_depth' : [6, 8, 10],
     'rf__min_samples_leaf' : [3, 5, 7]},
    {'rf__min_samples_leaf' : [3, 5, 7],
    'rf__min_samples_split': [3, 5, 10]}
]
#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

# model7 = SVC()
pipe = Pipeline([('mm',MinMaxScaler()), ('rf',RandomForestClassifier())])
# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=3, random_state=66)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1, random_state=66)
#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("걸린시간 : ", end - start)
print("RandomForestClassifier : ", result)
print("accuracy_score : ", acc)
'''
걸린시간 :  4.296555519104004
RandomForestClassifier :  1.0
accuracy_score :  1.0
'''