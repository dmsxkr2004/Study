# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸것

# m17_2 결과를 뛰어넘어랏!!!

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
# import warnings
# warnings.filterwarnings['ignore']

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
     "max_depth":[4, 5, 6], "eval_metric":['merror']},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
     "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "eval_metric":['merror']},
    {"n_estimators" : [90, 110], "learning_rate":[0.1, 0.001, 0.5],
     "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
     "colsample_bylevel":[0.6, 0.7, 0.9], "eval_metric":['merror']}
]
n_jobs = -1


#실습!!!
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print()
# # 케이폴드
# n_splits = 3
# kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 66)

# # 어펜드로 트레인 테스트 합치기

# x = np.append(x_train, x_test, axis = 0)
# y = np.append(y_train, y_test, axis = 0)

# print(x.shape)
# print(y.shape)

# # 카테고리컬
# # y = to_categorical(y)
# # 리쉐이프
# x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) # (70000, 786)

# print(x.shape)

# # 스케일러
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# # lda
# lda = LinearDiscriminantAnalysis(n_components = 9)#n_components = 컬럼의 갯수를 의미한다.
# x = lda.fit_transform(x,y)
# print(x.shape)

# # pca_EVR = lda.explained_variance_ratio_
# # print(pca_EVR)
# # print(sum(pca_EVR))

# # cumsum = np.cumsum(pca_EVR)
# # print(np.argmax(cumsum>= 0.95)+1) # 154
# # print(np.argmax(cumsum>= 0.99)+1) # 331 
# # print(np.argmax(cumsum>= 0.999)+1) # 487
# # print(np.argmax(cumsum) +1) # 713
# x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 66, shuffle = True, train_size = 0.8) 
# model = RandomizedSearchCV(LGBMClassifier(use_label_encoder=False), parameters, cv=kfold, verbose=3, # CV = cross validation
#                      refit = True, random_state=66)

# #3. 훈련
# import time
# start = time.time()
# model.fit(x_train,y_train, eval_metric='error')
# end = time.time()
# #4. 평가, 예측

# # x_test = x_train # 과적합 상황 보여주기
# # y_test = y_train # train 데이터로 best_estimator_로 예측뒤 점수를 내면
#                    # best_score_ 나온다.

# print("최적의 매개변수 : ", model.best_estimator_) # 최적의 매개변수 :  SVC(C=1, kernel='linear')
# print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# print("best_score_ : ", model.best_score_) # best_score_ :  0.9916666666666668 # train 값에서 가장 좋은값 (훈련시킨 데이터)
# print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667

# y_predict = model.predict(x_test)
# print("accuacy_score : ", accuracy_score(y_test, y_predict)) # accuacy_score :  0.9666666666666667

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 ACC : ",accuracy_score(y_test,y_pred_best)) # 최적 튠 ACC :  0.9666666666666667

# print("걸린시간 : ", end - start)

# # # acc : 0.96
# '''
# 최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
#               colsample_bynode=1, colsample_bytree=0.6,
#               enable_categorical=False, eval_metric='merror', gamma=0,
#               gpu_id=-1, importance_type=None, interaction_constraints='',
#               learning_rate=0.5, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=110, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', use_label_encoder=False,
#               validate_parameters=1, ...)
# 최적의 파라미터 :  {'n_estimators': 110, 'max_depth': 6, 'learning_rate': 0.5, 'eval_metric': 'merror', 'colsample_bytree': 0.6, 'colsample_bylevel': 0.9}
# best_score_ :  0.9150714230962489
# model.score :  0.9155
# accuacy_score :  0.9155
# 최적 튠 ACC :  0.9155
# 걸린시간 :  413.61247658729553
# PS D:\Study> 
# '''
# '''
# 최적의 매개변수 :  LGBMClassifier(colsample_bylevel=0.9, colsample_bytree=0.9,
#                eval_metric='merror', max_depth=6, n_estimators=90,
#                use_label_encoder=False)
# 최적의 파라미터 :  {'n_estimators': 90, 'max_depth': 6, 'learning_rate': 0.1, 'eval_metric': 'merror', 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9}
# best_score_ :  0.9150357237980243
# model.score :  0.9147142857142857
# accuacy_score :  0.9147142857142857
# 최적 튠 ACC :  0.9147142857142857
# 걸린시간 :  26.64940357208252
# '''
b = []
a = input()
for i in range(1,10):
    for j in range(1, 10):
        print(i*j)