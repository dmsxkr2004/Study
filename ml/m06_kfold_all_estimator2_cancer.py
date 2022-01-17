import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
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

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter = 'classifier')
allAlgorithms = all_estimators(type_filter = 'classifier')
print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print("ACC : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))
    except:
        # continue
        print(name, '은 에러 터진놈!!!')
'''
ACC :  [0.91208791 0.94505495 0.9010989  0.85714286 0.91208791]
 cross_val_score :  0.9055
ACC :  [0.89010989 0.97802198 0.69230769 0.84615385 0.93406593] 
 cross_val_score :  0.8681
ACC :  [0.94505495 0.98901099 0.91208791 0.92307692 0.98901099]
 cross_val_score :  0.9516
RadiusNeighborsClassifier 은 에러 터진놈!!!
ACC :  [0.92307692 0.98901099 0.95604396 0.95604396 0.96703297] 
 cross_val_score :  0.9582
ACC :  [0.96703297 0.98901099 0.94505495 0.9010989  0.96703297] 
 cross_val_score :  0.9538
ACC :  [0.95604396 0.97802198 0.94505495 0.91208791 0.96703297] 
 cross_val_score :  0.9516
ACC :  [0.93406593 0.9010989  0.92307692 0.84615385 0.76923077]
 cross_val_score :  0.8747
ACC :  [0.87912088 0.97802198 0.92307692 0.89010989 0.9010989 ] 
 cross_val_score :  0.9143
'''