import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

# x = np.delete(x, (0, 1, 2, 3), axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)



#2. 모델구성
model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()


#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)

# 5. 스코어
print(model1.score(x_test, y_test))
print(model2.score(x_test, y_test))
print(model3.score(x_test, y_test))
print(model4.score(x_test, y_test))


'''
결과 비교
1. DecisionTree 
acc : -0.07424390444661944
컬럼 삭제 후 acc : 0.5865137878599562
2. RandomForest
acc : 0.31748188278058187
컬럼 삭제 후 acc : 0.8695681570085672
3. XGBoost
acc : 0.3303614208368697
컬럼 삭제 후 acc : 0.8171628455140707
4. GradientBoosting 
acc : 0.32684911215347945
컬럼 삭제 후 acc : 0.8568369938126872
'''