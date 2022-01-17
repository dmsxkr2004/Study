import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_iris()
x = datasets.data
y = datasets.target

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

model = SVC()

scores = cross_val_score(model, x, y, cv = kfold)
print("ACC : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))