import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_iris()
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
ACC :  [       nan 0.95833333 0.91666667 1.         0.875     ]
 cross_val_score :  nan
ACC :  [0.95833333 0.95833333 0.95833333 1.         0.875     ] 
 cross_val_score :  0.95
ACC :  [0.91666667 0.83333333 0.79166667 0.83333333 0.83333333]
 cross_val_score :  0.8417
ACC :  [0.91666667 0.83333333 0.79166667 0.83333333 0.83333333] 
 cross_val_score :  0.8417
ACC :  [0.75       0.79166667 0.58333333 0.70833333 0.75      ]
 cross_val_score :  0.7167
ACC :  [0.95833333 1.         0.95833333 1.         0.875     ] 
 cross_val_score :  0.9583
'''