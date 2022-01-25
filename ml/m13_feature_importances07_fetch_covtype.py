#13_1 번을 카피
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target
# x = pd.DataFrame(x)
# x = x.drop([0], axis=1)
# x.info()
# x.to_numpy()
# print(type(x))
# x = np.delete(x, 0, axis=1)
import tensorflow as tf

print(x.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

model1 = DecisionTreeClassifier(max_depth = 3, random_state=66)
model2 = RandomForestClassifier(max_depth = 3, random_state=66)
model3 = XGBClassifier()
model4 = GradientBoostingClassifier(random_state=66)

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 평가, 예측

result1 = model1.score(x_test, y_test)
result2 = model2.score(x_test, y_test)
result3 = model3.score(x_test, y_test)
result4 = model4.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
y_predict2 = model2.predict(x_test)
y_predict3 = model3.predict(x_test)
y_predict4 = model4.predict(x_test)

acc1 = accuracy_score(y_test, y_predict1)
acc2 = accuracy_score(y_test, y_predict2)
acc3 = accuracy_score(y_test, y_predict3)
acc4 = accuracy_score(y_test, y_predict4)

print("DecisionTreeClassifier : ", result1)
print("RandomForestClassifier : ", result2)
print("XGBClassifier : ", result3)
print("GradientBoostingClassifier : ", result4)

print("accuracy_score : ", acc1)
print("accuracy_score : ", acc2)
print("accuracy_score : ", acc3)
print("accuracy_score : ", acc4)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)
'''
DecisionTreeClassifier :  0.9
accuracy_score :  0.9
[0.         0.57385373 0.42614627]
'''
'''
RandomForestClassifier :  1.0
accuracy_score :  1.0
[0.14884697 0.40483696 0.44631606]
'''
'''
XGBClassifier :  0.9
accuracy_score :  0.9
[0.02876592 0.6337989  0.33743513]
'''
'''
GradientBoostingClassifier :  0.9333333333333333
accuracy_score :  0.9333333333333333
[0.01645295 0.29869528 0.68485177]
'''
import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Impotances")
    plt.ylabel("Fetures")
    plt.ylim(-1, n_features)

plt.subplot(2,2,1)
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2)
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3)
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4)
plot_feature_importances_dataset(model4)
# plot_feature_importances_dataset(model1,model2,model3,model4)
plt.show()
