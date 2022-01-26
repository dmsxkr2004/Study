#GridSearchCV 적용해서 출력한 값에서 피처 임포턴스추출후
#SelectFromModel 만들어서 컬럼 축소 후 모델 구축해서 결과 도출
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_diabetes,load_boston,fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MaxAbsScaler, MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(datasets.feature_names)
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
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 66)
parameters = [
    {'n_estimators' : [1000, 5000, 10000],'max_depth' : [3, 4, 5, 6, 7]},
    {'learning_rate' : [0.05, 0.08, 0.1], 'colsample_bytree' : [0.7, 0.8, 0.9, 1]},
    {'min_child_weight' : [0.8, 0.9, 1],'subsample' : [1]},
    # {'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor', 'gpu_id' : 0}
]
#2. 모델
# model = XGBClassifier(
#     # n_jobs = -1,
#     random_state = 66,
#     n_estimators = 10000, # ephocs 랑 같은 역할을 함
#     learning_rate = 0.085, # 0.9412668531295966
#     max_depth = 5,
#     min_child_weight = 1,
#     subsample = 1,
#     colsample_bytree = 0.8,
#     reg_alpha = 1,          # 규제 L1
#     reg_lamda = 0,          # 규제 L2
#     tree_method = 'gpu_hist',
#     predictor = 'gpu_predictor',
#     gpu_id=0,
# )
model = GridSearchCV(XGBClassifier(tree_method = 'gpu_hist',
                                   predictor = 'gpu_predictor',
                                   gpu_id=0,), parameters, cv=kfold, refit = True)
#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test, y_test)],
          eval_metric='merror',        # rmse, mae, logloss, error
          early_stopping_rounds=100
          )
end = time.time()

print("걸린시간 : ", end-start)

#4. 평가,예측
print("최적의 매개변수 : ", model.best_estimator_) # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_) # best_score_ :  0.9916666666666668 # train 값에서 가장 좋은값 (훈련시킨 데이터)
print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667

y_predict = model.predict(x_test)
print("accuacy_score : ", accuracy_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ",accuracy_score(y_test,y_pred_best))

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
thresholds = np.sort(model.feature_importances_)
print("==================================")
'''
[0.05763175 0.00522148 0.00324451 0.00964329 0.00546137 0.00963201
 0.00537119 0.00743128 0.00368336 0.00847872 0.06420177 0.02843555
 0.02725693 0.06339701 0.00332207 0.03831341 0.01525623 0.04005617
 0.00484752 0.00495998 0.00182482 0.00572131 0.00775171 0.02193595
 0.00601    0.04227085 0.01156628 0.00570024 0.00062806 0.00535626
 0.01191347 0.00968608 0.00771433 0.00745034 0.0265401  0.05177546
 0.0289546  0.01679794 0.01430419 0.00463533 0.0172008  0.00571756
 0.01971706 0.01840449 0.01816438 0.0227764  0.01396742 0.00832503
 0.01364586 0.00356779 0.0392964  0.03674302 0.0602136  0.02787717]
[0.00062806 0.00182482 0.00324451 0.00332207 0.00356779 0.00368336
 0.00463533 0.00484752 0.00495998 0.00522148 0.00535626 0.00537119
 0.00546137 0.00570024 0.00571756 0.00572131 0.00601    0.00743128
 0.00745034 0.00771433 0.00775171 0.00832503 0.00847872 0.00963201
 0.00964329 0.00968608 0.01156628 0.01191347 0.01364586 0.01396742
 0.01430419 0.01525623 0.01679794 0.0172008  0.01816438 0.01840449
 0.01971706 0.02193595 0.0227764  0.0265401  0.02725693 0.02787717
 0.02843555 0.0289546  0.03674302 0.03831341 0.0392964  0.04005617
 0.04227085 0.05177546 0.05763175 0.0602136  0.06339701 0.06420177]
'''
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh,
                                prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(tree_method = 'gpu_hist',
                                    predictor = 'gpu_predictor',
                                    gpu_id=0,
                                    )
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))


'''
(464809, 54) (116203, 54)
Thresh=0.001, n=54, R2: 73.62%
(464809, 53) (116203, 53)
Thresh=0.002, n=53, R2: 74.31%
(464809, 52) (116203, 52)
Thresh=0.003, n=52, R2: 73.92%
(464809, 51) (116203, 51)
Thresh=0.003, n=51, R2: 74.43%
(464809, 50) (116203, 50)
Thresh=0.004, n=50, R2: 75.00%
(464809, 49) (116203, 49)
Thresh=0.004, n=49, R2: 74.08%
(464809, 48) (116203, 48)
Thresh=0.005, n=48, R2: 74.34%
(464809, 47) (116203, 47)
Thresh=0.005, n=47, R2: 74.77%
(464809, 46) (116203, 46)
Thresh=0.005, n=46, R2: 74.76%
(464809, 45) (116203, 45)
Thresh=0.005, n=45, R2: 74.39%
(464809, 44) (116203, 44)
Thresh=0.005, n=44, R2: 74.22%
(464809, 43) (116203, 43)
Thresh=0.005, n=43, R2: 74.18%
(464809, 42) (116203, 42)
Thresh=0.005, n=42, R2: 73.47%
(464809, 41) (116203, 41)
Thresh=0.006, n=41, R2: 72.63%
(464809, 40) (116203, 40)
Thresh=0.006, n=40, R2: 73.44%
(464809, 39) (116203, 39)
Thresh=0.006, n=39, R2: 73.04%
(464809, 38) (116203, 38)
Thresh=0.006, n=38, R2: 72.99%
(464809, 37) (116203, 37)
Thresh=0.007, n=37, R2: 73.62%
(464809, 36) (116203, 36)
Thresh=0.007, n=36, R2: 72.01%
(464809, 35) (116203, 35)
Thresh=0.008, n=35, R2: 72.00%
(464809, 34) (116203, 34)
Thresh=0.008, n=34, R2: 72.40%
(464809, 33) (116203, 33)
Thresh=0.008, n=33, R2: 72.07%
(464809, 32) (116203, 32)
Thresh=0.008, n=32, R2: 72.32%
(464809, 31) (116203, 31)
Thresh=0.010, n=31, R2: 64.81%
(464809, 30) (116203, 30)
Thresh=0.010, n=30, R2: 54.01%
(464809, 29) (116203, 29)
Thresh=0.010, n=29, R2: 47.94%
(464809, 28) (116203, 28)
Thresh=0.012, n=28, R2: 47.93%
(464809, 27) (116203, 27)
Thresh=0.012, n=27, R2: 47.84%
(464809, 26) (116203, 26)
Thresh=0.014, n=26, R2: 47.51%
(464809, 25) (116203, 25)
Thresh=0.014, n=25, R2: 47.01%
(464809, 24) (116203, 24)
Thresh=0.014, n=24, R2: 46.58%
(464809, 23) (116203, 23)
Thresh=0.015, n=23, R2: 46.61%
(464809, 22) (116203, 22)
Thresh=0.017, n=22, R2: 46.47%
(464809, 21) (116203, 21)
Thresh=0.017, n=21, R2: 46.30%
(464809, 20) (116203, 20)
Thresh=0.018, n=20, R2: 46.18%
(464809, 19) (116203, 19)
Thresh=0.018, n=19, R2: 45.91%
(464809, 18) (116203, 18)
Thresh=0.020, n=18, R2: 45.67%
(464809, 17) (116203, 17)
Thresh=0.022, n=17, R2: 45.61%
(464809, 16) (116203, 16)
Thresh=0.023, n=16, R2: 45.33%
(464809, 15) (116203, 15)
Thresh=0.027, n=15, R2: 44.36%
(464809, 14) (116203, 14)
Thresh=0.027, n=14, R2: 44.33%
(464809, 13) (116203, 13)
Thresh=0.028, n=13, R2: 44.34%
(464809, 12) (116203, 12)
Thresh=0.028, n=12, R2: 42.65%
(464809, 11) (116203, 11)
Thresh=0.029, n=11, R2: 40.95%
(464809, 10) (116203, 10)
Thresh=0.037, n=10, R2: 40.50%
(464809, 9) (116203, 9)
Thresh=0.038, n=9, R2: 37.95%
(464809, 8) (116203, 8)
Thresh=0.039, n=8, R2: 37.60%
(464809, 7) (116203, 7)
Thresh=0.040, n=7, R2: 37.25%
(464809, 6) (116203, 6)
Thresh=0.042, n=6, R2: 36.98%
(464809, 5) (116203, 5)
Thresh=0.052, n=5, R2: 36.84%
(464809, 4) (116203, 4)
Thresh=0.058, n=4, R2: 36.31%
(464809, 3) (116203, 3)
Thresh=0.060, n=3, R2: 17.17%
(464809, 2) (116203, 2)
Thresh=0.063, n=2, R2: 12.13%
(464809, 1) (116203, 1)
Thresh=0.064, n=1, R2: 4.17%
'''

# y_predict = model.predict(x_test)
# print('r2_score : ', r2_score(y_test, y_predict))
'''
걸린시간 :  305.636066198349
결과 :  0.9103035205631524
'''