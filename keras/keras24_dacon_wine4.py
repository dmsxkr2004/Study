from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn.metrics import accuracy_score


#1 데이터
path = "../_data/dacon/wine/" 
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv")

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['id', 'quality'], axis =1) # , 'pH', 'free sulfur dioxide', 'residual sugar'
print(x.shape)
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

print(x)                          # type column의 white, red를 0,1로 변환
print(x.shape)                    # (3231, 12)

from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))

test_file = test_file.drop(['id'], axis=1) # , 'pH', 'free sulfur dioxide', 'residual sugar'
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']
# print(y.unique())                # [6 7 5 8 4]
# y = get_dummies(y)
# print(y)                         #        4  5  6  7  8
                                   #  0     0  0  1  0  0
                                   #  1     0  0  0  1  0
                                   #  2     0  0  1  0  0
                                   #  3     0  1  0  0  0
                                   #  4     0  0  0  1  0

# # y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 123)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

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

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# # print(datetime)

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
# model_path = "".join([filepath, 't24_', datetime, '_', filename])
# es = EarlyStopping(monitor='accuracy', patience=200, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="accuracy", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

ensemble_clf = VotingClassifier(estimators = [('model1', model1), 
                                              ('model2', model2), 
                                              ('model3', model3)], voting = 'soft')
ensemble_clf.fit(x_train, y_train)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es, mcp])

# model = load_model("./_ModelCheckPoint/t24_1208_1314_0043-0.6014.hdf5")

#4 평가예측
# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss[0])                      # List 형태로 제공된다
# print("accuracy : ",loss[1])

################################ 제출용 ########################################
y_pred = ensemble_clf.predict(test_file)
# print('Test accuracy:', accuracy_score(y_pred, y_test))
submission['quality'] = y_pred
submission.to_csv(path + "jechul18.csv", index = False)


# result = model.predict(test_file)
# print(result[:5])
# result_recover = np.argmax(result, axis=1).reshape(-1,1) + 4
# result_recover = np.argmax(y_pred, axis=1).reshape(-1,1) + 4
# print(result_recover[:5])
# print(np.unique(result_recover))                           # value_counts = pandas에서만 먹힌다. 
# submission['quality'] = result_recover
# print(submission[:10])
# submission.to_csv(path + "mcp11.csv", index = False)
# print(result_recover)


