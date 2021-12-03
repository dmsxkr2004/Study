import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.python.keras.backend import relu

#1 데이터

path = "../_data/dacon/wine/" 
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path + "sample_Submission.csv") #제출할 값

x = train.drop(['quality'], axis =1) #
y = train['quality']
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()
le.fit(train.type)
x_type = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
x['type'] = x_type
print(x.type.value_counts())
#y = to_categorical(y) 
print(y.shape)
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #


scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 13))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(9, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # categorical_crossentorpy 결과값이[0~3]이상일때 적합한 값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀 
                                                                                        # # 리스트[]는 다른 값이 더 들어갈수도 있음
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 32, 
                 validation_split = 0.2 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

le.fit(test_flie.type)
test_flie_type = le.transform(test_flie['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
test_flie['type'] = test_flie_type
# y_predict = model.predict(x_test)

print(test_flie)

scaler.transform(test_flie)
result = model.predict(test_flie)
print(result[:5])
result_recover = np.argmax(result, axis =1).reshape(-1,1)
print(result_recover[:5])
# print(result_recover.value_counts()) # value_counts = pandas에서만 먹힌다.
print(np.unique(result_recover))
submission['quality'] = result_recover

# print(submission[:10])
submission.to_csv(path+"jechul1.csv", index = False)

'''
result = model.predict(test_flie)

result_recover = np.argmax(y, axis =1).reshape(-1,1)

submission['quality'] = result_recover
'''
#argumax
"""
'''
숙제
'''