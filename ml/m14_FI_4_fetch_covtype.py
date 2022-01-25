# 실습
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 칼럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

#2. 모델구성
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.datasets import fetch_covtype

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x = np.delete(x, (1, 2, 3, 4, 5, 26, 27, 28, 29, 30), axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

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
acc : 0.9397778026384861
컬럼 삭제 후 acc : 0.875424902971524
2. RandomForest
acc : 0.9556637952548557
컬럼 삭제 후 acc : 0.8968873437002487
3. XGBoost
acc : 0.869392356479609
컬럼 삭제 후 acc : 0.8074318218978856
4. GradientBoosting 
acc : 0.773499823584589
컬럼 삭제 후 acc : 0.7489221448671721
'''