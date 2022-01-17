import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True, random_state = 66, train_size=0.8)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

model1 = Perceptron()
model2 = KNeighborsClassifier()
model3 = LogisticRegression()
model4 = DecisionTreeClassifier()
model5 = RandomForestClassifier()
model6 = LinearSVC()
model7 = SVC()

scores = cross_val_score(model7, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))


'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
 cross_val_score :  0.9667
SVC :  0.9666666666666667
accuracy_score :  0.9666666666666667
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
 cross_val_score :  0.9667
Perceptron :  0.9666666666666667
accuracy_score :  0.9666666666666667
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
 cross_val_score :  0.9667
KNeighborsClassifier :  0.9666666666666667
accuracy_score :  0.9666666666666667
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
 cross_val_score :  0.9667
LogisticRegression :  1.0
accuracy_score :  1.0
'''