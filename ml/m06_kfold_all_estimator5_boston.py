import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_boston()
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
r2 :  [0.8718593  0.7278242  0.8019256  0.86264908 0.88711689] 
 cross_val_score :  0.8303
RegressorChain 은 에러 터진놈!!!
r2 :  [0.60059758 0.69676082 0.64998519 0.76759578 0.69030006]
 cross_val_score :  0.681
r2 :  [0.58601578 0.69889696 0.65394704 0.77347992 0.70039336]
 cross_val_score :  0.6825
r2 :  [-2.42063697e+26 -1.19959026e+27 -3.08729585e+26 -2.54392675e+26
 -1.46983105e+25]
 cross_val_score :  -4.038949063163179e+26
r2 :  [0.01715295 0.28593175 0.22882816 0.19290368 0.1284099 ] 
 cross_val_score :  0.1706
StackingRegressor 은 에러 터진놈!!!
r2 :  [0.53360851 0.6598974  0.59221485 0.74863007 0.61795529] 
 cross_val_score :  0.6305
r2 :  [0.5815212  0.69885237 0.6537276  0.77449543 0.70223459]
 cross_val_score :  0.6822
'''
