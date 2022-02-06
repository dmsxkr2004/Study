import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import fetch_california_housing,load_boston,fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold
from sklearn.metrics import r2_score,accuracy_score, f1_score
import pickle
path = "./_save/"
datasets = pickle.load(open(path+'m26_pickle1_save.dat','rb')) # 넘파이로 바꿔서 저장했을때는 컬럼과 인덱스가 저장이 안됨 ,
                                                               # 하지만 피클로 저장할 경우엔 인덱스와 컬럼이 맞게 저장돼서 저장함

x = datasets.data
y = datasets['target']

x = pd.DataFrame(x)

feature_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 
                  'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 
                  'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 
                  'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 
                  'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 
                  'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 
                  'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 
                  'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']

x.columns = feature_names

x = x.drop(['Hillshade_3pm','Soil_Type_6', 'Slope', 'Soil_Type_0'], axis=1)
x = x.to_numpy()
x_train,x_test,y_train,y_test = train_test_split(x, y, 
                                                 random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=0, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=9, min_child_weight=1,
              monotone_constraints='()', n_estimators=5000,
              num_parallel_tree=1, objective='multi:softprob',
              predictor='gpu_predictor', random_state=66, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)


#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='mlogloss',        # rmse, mae, logloss, error
          early_stopping_rounds=100
          )
end = time.time()

print("걸린시간 : ", end-start)

#4. 평가

results = model.score(x_test, y_test)
print("result : ", round(results,4))

y_predict = model.predict(x_test)
f1 = f1_score(y_test,y_predict,average='macro')
f2 = f1_score(y_test,y_predict,average='micro')
print("f1_score : ", f1)
print("f1_score : ", f2)


'''
걸린시간 :  7.180659055709839
result :  0.8642
r2 :  0.8642
'''

print("=====================================================")
hist = model.evals_result()
print(results)


path = './_save/'
# pickle.dump(model, open(path + 'm23_pickle1_save.dat','wb'))
model.save_model(path + "fetch_save3.dat")