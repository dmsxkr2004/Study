# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'kernel': 'linear', 'degree': 3, 'C': 1}

from inspect import Parameter
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True, random_state = 66, train_size=0.8)

#2. 모델구성
model = SVC(C=1, kernel='linear',degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, # CV = cross validation
#                      refit = True)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측

print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667

y_predict = model.predict(x_test)
print("accuacy_score : ", accuracy_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667
