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
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x = np.delete(x, (0, 6, 9), axis=1)

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
acc : -0.21397419564924358
컬럼 삭제 후 acc : 0.04771805186041511
2. RandomForest
acc : 0.37167366432518456
컬럼 삭제 후 acc : 0.34703394304656443
3. XGBoost
acc : 0.23802704693460175
컬럼 삭제 후 acc : 0.31285481131017956
4. GradientBoosting 
acc : 0.38999399452945527
컬럼 삭제 후 acc : 0.34855506724585583
'''
