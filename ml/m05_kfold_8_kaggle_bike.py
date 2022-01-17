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
from sklearn.model_selection import train_test_split,KFold,cross_val_score
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
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

model1 = Perceptron()
model2 = KNeighborsRegressor()
model3 = LinearRegression()
model4 = DecisionTreeRegressor()
model5 = RandomForestRegressor()
model6 = LinearSVC()
model7 = SVC()

scores = cross_val_score(model7, x_train, y_train, cv = kfold)
print("r2 : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))

'''
ACC :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] # Perceptron
 cross_val_score :  0.78
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] # KNeighborsClassifier
 cross_val_score :  0.96
'''
'''
ACC :  [1.         0.96666667 1.         0.9        0.96666667] # LogisticRegression
 cross_val_score :  0.9667
'''
'''
ACC :  [0.93333333 0.96666667 1.         0.9        0.93333333] # DecisionTreeClassifier
 cross_val_score :  0.9467
'''
'''
ACC :  [0.93333333 0.96666667 1.         0.9        0.96666667] # RandomForestClassifier
 cross_val_score :  0.9533
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.9        1.        ] # LinearSVC
 cross_val_score :  0.9667
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] # SVC
 cross_val_score :  0.9667
'''