import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
import sklearn as sk
warnings.filterwarnings('ignore')
#1. 데이터

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape) # (506, 13) -> (20640, 8)

pca = PCA(n_components = 13)#n_components = 컬럼의 갯수를 의미한다.
x = pca.fit_transform(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 66, shuffle = True)

#2. 모델
from xgboost import XGBRegressor,XGBClassifier

model = XGBClassifier()

#3. 훈련
model.fit(x_train,y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)
'''
(569, 30)
결과 :  0.9736842105263158
'''
