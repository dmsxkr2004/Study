import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True, random_state = 66, train_size=0.8)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter = 'classifier')
allAlgorithms = all_estimators(type_filter = 'regressor')
print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print("r2 : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))
    except:
        # continue
        print(name, '은 에러 터진놈!!!')
'''
r2 :  [0.53678693 0.47844048 0.47456644 0.56014497 0.40329553] 
 cross_val_score :  0.4906
r2 :  [0.47793293 0.52818634 0.44317229 0.52822598 0.35927781] 
 cross_val_score :  0.4674
r2 :  [0.40016088 0.48592609 0.43362971 0.54818166 0.39952648] 
 cross_val_score :  0.4535
r2 :  [0.52920538 0.48972847 0.47272996 0.54527576 0.39899837]
 cross_val_score :  0.4872
'''
