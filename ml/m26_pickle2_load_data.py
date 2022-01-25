import time
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import fetch_california_housing,load_boston,fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,accuracy_score

# import warnings
# warnings.filterwarnings('ignore')

# #1. 데이터
# datasets = fetch_covtype()
# x = datasets.data
# y = datasets['target']
# print(x.shape, y.shape) # (20640, 8) (20640,)

import pickle
path = "./_save/"
datasets = pickle.load(open(path+'m26_pickle1_save.dat','rb')) # 넘파이로 바꿔서 저장했을때는 컬럼과 인덱스가 저장이 안됨 ,
                                                               # 하지만 피클로 저장할 경우엔 인덱스와 컬럼이 맞게 저장돼서 저장함

x = datasets.data
y = datasets['target']
x_train,x_test,y_train,y_test = train_test_split(x, y, 
                                                 random_state=66, shuffle=True, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
    n_jobs = -1,
    n_estimators = 1000, # ephocs 랑 같은 역할을 함
    learning_rate = 0.085, # 0.9412668531295966
    max_depth = 3,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 0.8,
    reg_alpha = 1,          # 규제 L1
    reg_lamda = 0,          # 규제 L2
)

#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='mlogloss',        # rmse, mae, logloss, error
          early_stopping_rounds=10
          )
end = time.time()

print("걸린시간 : ", end-start)

#4. 평가

results = model.score(x_test, y_test)
print("result : ", round(results,4))

y_pred = model.predict(x_test)
r2 = accuracy_score(y_test, y_pred)
print("accuracy_score : ", round(r2,4))

# 0.843

'''
걸린시간 :  7.180659055709839
result :  0.8642
r2 :  0.8642
'''

print("=====================================================")
hist = model.evals_result()
print(results)


pickle.dump(model, open(path + 'm23_pickle1_save.dat','wb'))