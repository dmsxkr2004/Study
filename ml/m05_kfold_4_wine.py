import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_wine()
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
ACC :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] # Perceptron
 cross_val_score :  0.78
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] # KNeighborsClassifier
 cross_val_score :  0.96
'''
'''
ACC :  [1.         0.96666667 1.         0.9        0.96666667] # LogisticRegression
 cross_val_score :  0.9667
'''
'''
ACC :  [0.93333333 0.96666667 1.         0.9        0.93333333] # DecisionTreeClassifier
 cross_val_score :  0.9467
'''
'''
ACC :  [0.93333333 0.96666667 1.         0.9        0.96666667] # RandomForestClassifier
 cross_val_score :  0.9533
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.9        1.        ] # LinearSVC
 cross_val_score :  0.9667
'''
'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] # SVC
 cross_val_score :  0.9667
'''