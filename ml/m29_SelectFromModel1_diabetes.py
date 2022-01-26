from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MaxAbsScaler, MinMaxScaler,PowerTransformer,PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)
# x = pd.DataFrame(x, columns=datasets['feature_names'])
# print(x.feature_names)
x = pd.DataFrame(x)
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x.columns = feature_names
print(x)
x = x.drop(['age', 'sex', 's1', 's4'], axis=1)
x = x.to_numpy()
# x = pd.DataFrame(x)
# print(type(x))
# print(x.info())
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(
    #n_jobs = -1,
    n_estimators = 2000, # ephocs 랑 같은 역할을 함
    learning_rate = 0.085, # 0.9412668531295966
    max_depth = 3,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 0.8,
    reg_alpha = 1,          # 규제 L1
    reg_lamda = 0,          # 규제 L2
    ree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    gpu_id=0,
)
#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='mae',        # rmse, mae, logloss, error
          early_stopping_rounds=50
          )
end = time.time()

print("걸린시간 : ", end-start)
#4. 평가,예측
score = model.score(x_test,y_test)
print("결과 : ", score)

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
thresholds = np.sort(model.feature_importances_)
print("==================================")
'''
[0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
 0.06597802 0.07382318 0.19681741 0.39979857]
'''
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh,
                                prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(#n_jobs = -1
                                   ree_method = 'gpu_hist',
                                    predictor = 'gpu_predictor',
                                    gpu_id=0,)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))

'''
(353, 10) (89, 10)
Thresh=0.026, n=10, R2: 23.96%
(353, 9) (89, 9)
Thresh=0.033, n=9, R2: 27.03%
(353, 8) (89, 8)
Thresh=0.038, n=8, R2: 23.87%
(353, 7) (89, 7)
Thresh=0.048, n=7, R2: 26.48%
(353, 6) (89, 6)
Thresh=0.055, n=6, R2: 30.09%
(353, 5) (89, 5)
Thresh=0.063, n=5, R2: 27.41%
(353, 4) (89, 4)
Thresh=0.066, n=4, R2: 29.84%
(353, 3) (89, 3)
Thresh=0.074, n=3, R2: 23.88%
(353, 2) (89, 2)
Thresh=0.197, n=2, R2: 14.30%
(353, 1) (89, 1)
Thresh=0.400, n=1, R2: 2.56%
'''

# # y_predict = model.predict(x_test)
# # print('r2_score : ', r2_score(y_test, y_predict))

# 0.41