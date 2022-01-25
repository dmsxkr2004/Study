# 실습
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 칼럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

#2. 모델구성
import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x = np.delete(x, (1, 2, 3, 4), axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


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
acc : 0.8186830891113592
컬럼 삭제 후 acc : 0.5865137878599562
2. RandomForest
acc : 0.9214136541819238
컬럼 삭제 후 acc : 0.8695681570085672
3. XGBoost
acc : 0.9221188601856797
컬럼 삭제 후 acc : 0.8171628455140707
4. GradientBoosting 
acc : 0.9460785072249167
컬럼 삭제 후 acc : 0.8568369938126872
'''