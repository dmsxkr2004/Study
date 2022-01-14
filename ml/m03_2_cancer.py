from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류
from sklearn.linear_model import LogisticRegression, LinearRegression # 분류
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score


#1 데이터

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 123)


#2 모델구성

# from keras.layers import BatchNormalization
def mlp_model():
    model = Sequential()
    model.add(Dense(100, input_shape =(12, ), activation='relu'))
    model.add(Dense(130, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='softmax'))

#3. 컴파일, 훈련
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model

model = mlp_model()

# 서로 다른 모델을 3개 만들어 합친다
model1 = KerasClassifier(build_fn = mlp_model, epochs = 190, verbose = 1)
model1._estimator_type="classifier" 
model2 = KerasClassifier(build_fn = mlp_model, epochs = 190, verbose = 1)
model2._estimator_type="classifier"
model3 = KerasClassifier(build_fn = mlp_model, epochs = 190, verbose = 1)
model3._estimator_type="classifier"



ensemble_clf = VotingClassifier(estimators = [('model1', model1), 
                                              ('model2', model2), 
                                              ('model3', model3)], voting = 'soft')
ensemble_clf.fit(x_train, y_train)

ensemble_clf.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = ensemble_clf.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = ensemble_clf.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("ensemble_clf : ", result)
print("accuracy_score : ", acc)

