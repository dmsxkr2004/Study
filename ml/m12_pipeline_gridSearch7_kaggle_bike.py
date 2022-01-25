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
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV,RandomizedSearchCV, StratifiedKFold
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
pipe = Pipeline([('mm',MinMaxScaler()), ('rf',RandomForestRegressor())])
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

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)

print("걸린시간 : ", end - start)
print("RandomForestRegressor : ", result)
print("r2_score : ", acc)
'''
걸린시간 :  27.085517168045044
RandomForestRegressor :  0.35104994998048644
r2_score :  0.35104994998048644
'''