import time
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import fetch_california_housing,load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split

from sklearn.metrics import r2_score,accuracy_score
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train,x_test,y_train,y_test = train_test_split(x, y, 
                                                 random_state=66, shuffle=True, train_size=0.8)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBRegressor(
    n_jobs = -1,
    n_estimators = 2500, # ephocs 랑 같은 역할을 함
    learning_rate = 0.1 # 0.8556698122229669
)

#3. 훈련
start = time.time()
model.fit(x_train,y_train, verbose=1)
end = time.time()

print("걸린시간 : ", end-start)

#4. 평가

results = model.score(x_test, y_test)
print("result : ", results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

# 0.843