import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, fetch_covtype
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import warnings
import sklearn as sk
warnings.filterwarnings('ignore')

#1. 데이터
# datasets = load_wine()
# datasets = load_breast_cancer()
datasets = fetch_covtype()
# datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x.shape) # (506, 13) -> (20640, 8)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 66, shuffle = True, stratify = y)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("lda 전 : ",x.shape)

lda = LinearDiscriminantAnalysis() # 디폴트 라벨의갯수, 컬럼의갯수 보다 1개 작은거
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
# print(x.shape)
print("lda 후 : ",x_train.shape)


#2. 모델
from xgboost import XGBRegressor,XGBClassifier

model = XGBClassifier()

#3. 훈련
# model.fit(x_train,y_train, eval_metric='error')
# model.fit(x_train,y_train, eval_metric='merror')
import time
start = time.time()
model.fit(x_train, y_train, eval_metric='error') # 이진분류 , 다중분류
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)
print("걸린시간 : ", end - start)
# iris
'''
lda 전 :  (150, 4)
lda 후 :  (120, 2)
결과 :  1.0
'''
# cancer
'''
lda 전 :  (569, 30)
lda 후 :  (455, 1)
결과 :  0.9473684210526315
'''
# wine
'''
lda 전 :  (178, 13)
lda 후 :  (142, 2)
결과 :  1.0
걸린시간 :  0.05903148651123047
'''
# fetch covtype
'''
lda 전 :  (581012, 54)
lda 후 :  (464809, 6)
결과 :  0.7878109859470065
걸린시간 :  81.62100100517273
'''