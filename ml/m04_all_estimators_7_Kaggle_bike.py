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
from sklearn.model_selection import train_test_split
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

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter = 'classifier')
allAlgorithms = all_estimators(type_filter = 'regressor')
print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)
    except:
        continue
        # print(name, '은 에러 터진놈!!!')
'''
ARDRegression 의 정답률 :  0.2595330084243965
AdaBoostRegressor 의 정답률 :  0.20353517561694767
BaggingRegressor 의 정답률 :  0.2640391275012687
BayesianRidge 의 정답률 :  0.2600644277520331
CCA 의 정답률 :  0.005402723490054662
DecisionTreeRegressor 의 정답률 :  -0.10683410412027028
DummyRegressor 의 정답률 :  -0.00017313927999396128
ElasticNet 의 정답률 :  -0.00017313927999396128
ElasticNetCV 의 정답률 :  0.2600429761788744
ExtraTreeRegressor 의 정답률 :  -0.07007811271836739
ExtraTreesRegressor 의 정답률 :  0.22686628681448617
GammaRegressor 의 정답률 :  0.0329799804704195
GaussianProcessRegressor 의 정답률 :  -6.405980427548343
GradientBoostingRegressor 의 정답률 :  0.32149247005531656
HistGradientBoostingRegressor 의 정답률 :  0.34276427863079184
HuberRegressor 의 정답률 :  0.24640631504982102
KNeighborsRegressor 의 정답률 :  0.2895919876039582
KernelRidge 의 정답률 :  -0.047182799609482684
Lars 의 정답률 :  0.25995216484963324
LarsCV 의 정답률 :  0.2534291706598264
Lasso 의 정답률 :  -0.00017313927999396128
LassoCV 의 정답률 :  0.2599444585697248
LassoLars 의 정답률 :  -0.00017313927999396128
LassoLarsCV 의 정답률 :  0.25993756199484586
LassoLarsIC 의 정답률 :  0.25995216484963324
LinearRegression 의 정답률 :  0.25995216484963324
LinearSVR 의 정답률 :  0.2320027820776156
MLPRegressor 의 정답률 :  0.31147454697357335
NuSVR 의 정답률 :  0.30481156474098203
PLSCanonical 의 정답률 :  -0.5298357213928355
PLSRegression 의 정답률 :  0.25543438482412306
PassiveAggressiveRegressor 의 정답률 :  -1.7083585820856175
PoissonRegressor 의 정답률 :  0.10679895221086211
'''