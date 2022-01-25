import time
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import fetch_california_housing,load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,accuracy_score
# import warnings
# warnings.filterwarnings('ignore')
#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train,x_test,y_train,y_test = train_test_split(x, y, 
                                                 random_state=66, shuffle=True, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBRegressor(
    n_jobs = -1,
    n_estimators = 100, # ephocs 랑 같은 역할을 함
    learning_rate = 0.085, # 0.9412668531295966
    max_depth = 5,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 0.8,
    reg_alpha = 1,          # 규제 L1
    reg_lamda = 0,          # 규제 L2
    
)

#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train,y_train),(x_test,y_test)],
          eval_metric='rmse'        # rmse, mae, logloss, error
          )
end = time.time()

print("걸린시간 : ", end-start)

#4. 평가

results = model.score(x_test, y_test)
print("result : ", round(results,4))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,4))

# 0.843
'''
걸린시간 :  7.180659055709839
result :  0.8642
r2 :  0.8642
'''

print("=====================================================")
results = model.evals_result()
print(results)

import matplotlib.pyplot as plt

train_error = results['validation_0']['rmse']
test_error = results['validation_1']['rmse']

epoch = range(1, len(train_error)+1)
plt.plot(epoch, train_error, label = 'Train')
plt.plot(epoch, test_error, label = 'Test')
plt.ylabel('Classification Error')
plt.xlabel('Model Complexity (n_estimators)')
plt.legend()
plt.show()