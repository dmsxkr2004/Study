# 실습
# 회귀 데이터들 몽땅 집어넣고 LDA2와 동일하게 만드시오!!
# 보스턴, 다이아비티스, 캘리포니아!!!

import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import warnings
import sklearn as sk
warnings.filterwarnings('ignore')

#1. 데이터
# datasets = load_diabetes()
datasets = fetch_california_housing()
# datasets = load_boston()


x = datasets.data
y = datasets.target
y = np.round(y,0)
# print(x.shape) # (506, 13) -> (20640, 8)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 66, shuffle = True) #stratify=y.iloc[:,1])

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

model = XGBRegressor()

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
# boston
'''
lda 전 :  (506, 13)
lda 후 :  (404, 13)
결과 :  0.9016514433565562
걸린시간 :  0.10349082946777344
'''
# diabetes
'''
lda 전 :  (442, 10)
lda 후 :  (353, 10)
결과 :  0.313354229055848
걸린시간 :  0.08637213706970215
'''
'''
lda 전 :  (20640, 8)
lda 후 :  (16512, 5)
결과 :  0.6575931583547636
걸린시간 :  0.4898793697357178
'''