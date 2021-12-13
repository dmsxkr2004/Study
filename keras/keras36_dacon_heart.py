import pandas as pd
import numpy as np
import datetime
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

#1. 데이터
path = "../_data/dacon/Heart/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path+ "sample_submission.csv")
print(train.shape) #(151, 15)
print(test_file.shape) #(152, 14)
print(submission.shape) #(152, 2)

x = train.drop(['id','target'], axis = 1)
test_file = test_file.drop(['id'], axis = 1)
y = train['target']
# print(x.shape) # (151, 14)
# print(y.shape) # (151,)

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.8, shuffle = True, random_state = 66)
# 트레인 테스트 스플릿 형태를 변형하면 쉐이프도 변형된다.

print(x_train.shape)# (120, 13)
print(x_test.shape)# (31, 13)
print(y_train.shape)# (120,)
print(y_test.shape)# (31,)
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_test_file = scaler.transform(test_file)

#2. 모델구성
model = Sequential()
model.add(Dense(55, activation = 'relu', input_shape = (13,)))
model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(10))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'Heart_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience = 150, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True,
                      filepath = model_path)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/Heart_1213_1750_0145-0.5346.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)

f1 = f1_score(y_pred, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])
print('f1 스코어 : ', f1)

###################### 제출용 제작 ###############################
results = model.predict(test_file)
results = results.round(0).astype(int)

submission['target'] = results
submission.to_csv(path+"jechul.csv", index = False)

'''
# f1-score 를 사용하게 된 이유

f1-score를 사용하게 되면 sigmoid값을 출력할때
accuracy 로 출력하면 1의 비율이 90%이고 0의 비율이 10%일때
acc 가 1을 결과값으로 다 넣었을때 0.9라는 수치가 떠서 0의 데이터값이 모자라서 신뢰할수 없는 데이터가 된다
이를 골고루 분포해서 결과값을 보여주는 알고리즘이 f1-score이므로 f1-score를 쓴다

'''