import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

#1. 데이터
path = "D:/Study/_data/"
datasets = pd.read_csv(path + "winequality-white.csv", sep=';', index_col=None, header=0)

x = datasets.drop(['fixed acidity','pH','quality'],axis = 1)
y = datasets['quality']
############################################################################


############################################################################
for index, value in enumerate(y):
    if value == 9:
        y[index] = 8
    elif value == 8:
        y[index] = 8
    elif value == 7:
        y[index] = 7
    elif value == 6:
        y[index] = 6
    elif value == 5:
        y[index] = 5
    elif value == 4:
        y[index] = 4
    elif value == 3:
        y[index] = 3
    else:
        y[index] = 0
        
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66, stratify=y)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    # n_jobs = -1,
    random_state = 66,
    n_estimators = 10000, # ephocs 랑 같은 역할을 함
    learning_rate = 0.085, # 0.9412668531295966
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 0.8,
    reg_alpha = 1,          # 규제 L1
    reg_lamda = 0,          # 규제 L2
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    gpu_id=0,
)

#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='merror',        # rmse, mae, logloss, error
          early_stopping_rounds=100
          )
end = time.time()

print("걸린시간 : ", end-start) # 37.47783923149109

#4. 평가

results = model.score(x_test, y_test)
print("result : ", round(results,4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred, average = 'micro')
f2 = f1_score(y_test,y_pred, average = 'macro')
print("accuracy_score : ", round(acc,4))
print("f1_score : ",round(f1,4))
print("f1_score : ",round(f2,4))
# import matplotlib.pyplot as plt
# from xgboost.plotting import plot_importance
# plot_importance(model)
# plt.show()
print("==============SMOTE 적용===================")
start1 = time.time()
smote = SMOTE(random_state=66, k_neighbors = 1, )
x_train, y_train = smote.fit_resample(x_train, y_train)
end1 = time.time()
print("SMOTE 걸린시간 : ", end1-start1)

#2. 모델
model = XGBClassifier(
    # n_jobs = -1,
    random_state = 66,
    n_estimators = 10000, # ephocs 랑 같은 역할을 함
    learning_rate = 0.085, # 0.9412668531295966
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 0.8,
    reg_alpha = 1,          # 규제 L1
    reg_lamda = 0,          # 규제 L2
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    gpu_id=0,
)

#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='merror',        # rmse, mae, logloss, error
          early_stopping_rounds=100
          )
end = time.time()

print("걸린시간 : ", end-start) # 37.47783923149109

#4. 평가

results = model.score(x_test, y_test)
print("result : ", round(results,4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred, average = 'micro')
f2 = f1_score(y_test,y_pred, average = 'macro')
print("accuracy_score : ", round(acc,4))
print("f1_score : ",round(f1,4))
print("f1_score : ",round(f2,4))