from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.utils import all_estimators

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
#1. 데이터 
path = '../_data/kaggle/bike/'
train = pd.read_csv(path+'train.csv')
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
# print(submit_file.columns)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']
# 로그변환
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 42)

print(x_train.shape)  # (7620, 8)
print(x_test.shape)  # (3266, 8)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 66)
parameters = [
    {'n_estimators' : [100, 200, 400],'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [1, 3, 5, 7, 9, 10],'min_samples_split' : [2, 3, 5, 7, 9, 10]},
    # {'n_jobs' : [-1, 2, 4]}
]

# 파라미터 조합으로 2개이상 엮을것
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, # CV = cross validation
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
print("r2_score : ", r2_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 r2_score : ",r2_score(y_test,y_pred_best)) # 최적 튠 ACC :  0.9666666666666667

print("걸린시간 : ", end - start)
'''
Fitting 5 folds for each of 20 candidates, totalling 100 fits
최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_estimators=200)
최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 10}
best_score_ :  0.34527667525192063
model.score :  0.3537987691419532
r2_score :  0.3537987691419532
최적 튠 r2_score :  0.3537987691419532
걸린시간 :  69.16246438026428
'''