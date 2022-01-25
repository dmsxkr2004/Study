from inspect import Parameter
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, HalvingGridSearchCV # 랜덤 = 랜덤하게 돌려본 , 하빙 = 데이터의 일부만 돌려보고 최적의 데이터를 추출해서씀
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold, GridSearchCV # 그리드 = 전부 돌려보고 최적의 값을 찾음
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True, random_state = 66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)
parameters = [
    {'n_estimators' : [100, 200],'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10],'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4] }                   # 24 5 x 3 x 4 = 60
]                                                                       # 총 42개 12 + 6 + 24 = 42

#2. 모델구성
# model = SVC(C=1, kernel='linear',degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, # CV = cross validation
#                      refit = True, n_jobs=-1)
# model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, # CV = cross validation
#                      refit = True, n_jobs=-1, random_state=66, n_iter=20)
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, # CV = cross validation
                     refit = True, n_jobs=-1,random_state=66)

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
best_score_ :  0.9916666666666668  # train 값에서 가장 좋은값 (훈련시킨 데이터)
model.score :  0.9666666666666667 # test 값에서 가장 좋은값 (훈련시키지 않은 데이터)
accuacy_score :  0.9666666666666667
걸린시간 :  0.7954084873199463
'''
###############################################################################
"""
print(model.cv_results_)# dict형으로나옴 (딕셔너리) std = 표준편차

aaa = pd.DataFrame(model.cv_results_)
print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
    #  'split0_test_score','split1_test_score','split2_test_score',
    #  'split3_test_score','split4_test_score'
    #  ]]
print(bbb)
'''
0           {'C': 1, 'degree': 3, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
1           {'C': 1, 'degree': 4, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
2           {'C': 1, 'degree': 5, 'kernel': 'linear'}         0.991667                1           1.000000           1.000000           0.958333           1.000000           1.000000
'''
"""
"""
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 27
n_resources: 20
Fitting 5 folds for each of 27 candidates, totalling 135 fits
----------
iter: 1
n_candidates: 9
n_resources: 60
Fitting 5 folds for each of 9 candidates, totalling 45 fits
----------
iter: 2
n_candidates: 3
n_resources: 180
Fitting 5 folds for each of 3 candidates, totalling 15 fits
최적의 매개변수 :  RandomForestClassifier(max_depth=10)
최적의 파라미터 :  {'max_depth': 10, 'n_estimators': 100}
best_score_ :  0.961111111111111
model.score :  0.9736842105263158
accuacy_score :  0.9736842105263158
최적 튠 ACC :  0.9736842105263158
걸린시간 :  3.984259843826294
"""