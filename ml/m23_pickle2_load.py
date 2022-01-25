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
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train,x_test,y_train,y_test = train_test_split(x, y, 
                                                 random_state=66, shuffle=True, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 불러오기
import pickle
path = './_save/'
model = pickle.load(open(path + 'm23_pickle1_save.dat', 'rb'))

#4. 평가
results = model.score(x_test,y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2_score : ",r2)

print("=====================================================")
hist = model.evals_result()
print(results)
'''
results :  0.9366541297748481
r2_score :  0.9366541297748481
=====================================================
0.9366541297748481
'''