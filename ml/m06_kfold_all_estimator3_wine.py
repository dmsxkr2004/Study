import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
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
ACC :  [0.93103448 0.62068966 0.78571429 0.32142857 0.25      ] 
 cross_val_score :  0.5818
MultiOutputClassifier 은 에러 터진놈!!!
ACC :  [0.93103448 0.89655172 0.82142857 0.85714286 0.71428571] 
 cross_val_score :  0.8441
ACC :  [0.86206897 0.86206897 0.57142857 0.71428571 0.67857143]
 cross_val_score :  0.7377
ACC :  [0.93103448 0.96551724 0.71428571 0.96428571 0.85714286] 
 cross_val_score :  0.8865
OneVsOneClassifier 은 에러 터진놈!!!
OneVsRestClassifier 은 에러 터진놈!!!
OutputCodeClassifier 은 에러 터진놈!!!
ACC :  [0.55172414 0.68965517 0.60714286 0.42857143 0.28571429] 
 cross_val_score :  0.5126
ACC :  [0.55172414 0.62068966 0.53571429 0.35714286 0.60714286]
 cross_val_score :  0.5345
ACC :  [0.96551724 1.         0.92857143 1.         1.        ] 
 cross_val_score :  0.9788
RadiusNeighborsClassifier 은 에러 터진놈!!!
ACC :  [0.96551724 1.         0.96428571 0.96428571 0.96428571] 
 cross_val_score :  0.9717
ACC :  [1. 1. 1. 1. 1.]
 cross_val_score :  1.0
ACC :  [1. 1. 1. 1. 1.] 
 cross_val_score :  1.0
ACC :  [0.79310345 0.51724138 0.53571429 0.82142857 0.5       ] 
 cross_val_score :  0.6335
ACC :  [0.5862069  0.65517241 0.5        0.67857143 0.67857143]
 cross_val_score :  0.6197
'''
