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

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)
# parameters = [
#     {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},    # 12
#     {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001,0.0001]},       # 6
#     {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],
#      "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                     # 24
# ]                                                                       # 총 42개
#2. 모델구성
model = SVC(C=1, kernel='linear',degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, # CV = cross validation
#                      refit = True)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측

# x_test = x_train # 과적합 상황 보여주기
# y_test = y_train # train 데이터로 best_estimator_로 예측뒤 점수를 내면
                   # best_score_ 나온다.

# print("최적의 매개변수 : ", model.best_estimator_) # 최적의 매개변수 :  SVC(C=1, kernel='linear')
# print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# print("best_score_ : ", model.best_score_) # best_score_ :  0.9916666666666668 # train 값에서 가장 좋은값 (훈련시킨 데이터)
print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667

y_predict = model.predict(x_test)
print("accuacy_score : ", accuracy_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 ACC : ",accuracy_score(y_test,y_pred_best)) # 최적 튠 ACC :  0.9666666666666667

'''
best_score_ :  0.9916666666666668  # train 값에서 가장 좋은값 (훈련시킨 데이터)
model.score :  0.9666666666666667 # test 값에서 가장 좋은값 (훈련시키지 않은 데이터)
accuacy_score :  0.9666666666666667
'''
###############################################################################
# print(model.cv_results_)# dict형으로나옴 (딕셔너리) std = 표준편차

# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
#     #  'split0_test_score','split1_test_score','split2_test_score',
#     #  'split3_test_score','split4_test_score'
#     #  ]]
# print(bbb)
'''
0           {'C': 1, 'degree': 3, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
1           {'C': 1, 'degree': 4, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
2           {'C': 1, 'degree': 5, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
'''