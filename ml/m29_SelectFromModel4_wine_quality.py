#GridSearchCV 적용해서 출력한 값에서 피처 임포턴스추출후
#SelectFromModel 만들어서 컬럼 축소 후 모델 구축해서 결과 도출
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer,QuantileTransformer # PowerTransformer : 이상치 제거 스케일러
from sklearn.preprocessing import PolynomialFeatures # degree 옵션으로 차수를 조절한다. x를 fit transform하여 새로운 데이터를 생성한다.
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
path = "D:/Study/_data/"
datasets = pd.read_csv(path + "winequality-white.csv", sep=';', index_col=None, header=0)

x = datasets.drop(['fixed acidity','pH','quality'],axis = 1)
y = datasets['quality']
# print(x.columns)
'''
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'], dtype='object')
'''
print(x.shape,y.shape) # (4898, 11) (4898,)
# le = LabelEncoder()
# label = x['type']
# le.fit(label)
# x['type'] = le.transform(label)

# y = get_dummies(y)
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66, stratify=y)
 
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
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
[0.04677639 0.14301346 0.06103746 0.07667924 0.05777257 0.08961623
 0.05154333 0.10763792 0.04860649 0.05641407 0.26090285]
[0.04677639 0.04860649 0.05154333 0.05641407 0.05777257 0.06103746
 0.07667924 0.08961623 0.10763792 0.14301346 0.26090285]
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
Thresh=0.049, n=10, R2: 66.73% 
'''
 
