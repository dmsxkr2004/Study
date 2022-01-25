#실습

#모델 : RandomForestClassifier
from inspect import Parameter
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 66)
parameters = [
    {'n_estimators' : [100, 200],'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10],'min_samples_split' : [2, 3, 5, 10]},
    # {'n_jobs' : [-1, 2, 4]}
]

# 파라미터 조합으로 2개이상 엮을것
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, # CV = cross validation
                     refit = True, random_state=66, n_iter=20)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()
#4. 평가, 예측

# x_test = x_train # 과적합 상황 보여주기
# y_test = y_train # train 데이터로 best_estimator_로 예측뒤 점수를 내면
                   # best_score_ 나온다.

print("최적의 매개변수 : ", model.best_estimator_) # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_) # best_score_ :  0.9916666666666668 # train 값에서 가장 좋은값 (훈련시킨 데이터)
print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667

y_predict = model.predict(x_test)
print("accuacy_score : ", accuracy_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ",accuracy_score(y_test,y_pred_best)) # 최적 튠 ACC :  0.9666666666666667

print("걸린시간 : ", end - start)
'''
Fitting 5 folds for each of 20 candidates, totalling 100 fits
최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5, min_samples_split=5)
최적의 파라미터 :  {'min_samples_split': 5, 'min_samples_leaf': 5}
best_score_ :  0.9583333333333334
model.score :  0.9333333333333333
accuacy_score :  0.9333333333333333
최적 튠 ACC :  0.9333333333333333
걸린시간 :  9.484269380569458
'''