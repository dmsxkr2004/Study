# 숙제 : 중위값 평균값 비교분석
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터
path = "./_data/bike/"# .은 지금(현재 작업폴더) 스터디 폴더를 의미한다.
train =  pd.read_csv(path + "train.csv")# csv = 엑셀파일과 같다 , # pd.read_csv('경로/파일명.csv')
test_file =  pd.read_csv(path + "test.csv") 
sampleSubmission_file = pd.read_csv(path + "sampleSubmission.csv")
#print(sampleSubmission_file.columns)
# print(train.shape) # (10886, 12)
# print(test.shape) # (6493, 9)
# print(sampleSubmission.shape) # (6493, 2)
#print(type(train))
#print(train.info())
#print(train.describe()) # std = 표준편차 min = 최소값 max = 최대값
#print(train.columns) 
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
#print(train.head())
#print(train.tail())

x = train.drop(['datetime','casual','registered','count'], axis = 1) # drop 리스트를 없애겠다. # axis
test_file = test_file.drop(['datetime'], axis = 1)
#print(x.columns)
#print(x.shape) # (10886, 8)
y = train['count']
#print(y)
print(y.shape) # (10886, )
y = np.log1p(y) # 로그변환
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.8, shuffle = True , random_state= 63)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 8))# linear= 기본값
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer ='adam')
#es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
model.fit(x_train, y_train, epochs=30, batch_size=50, 
          verbose = 1, validation_split = 0.1) #callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)

plt.plot(y)
plt.show()



# 로그 변환하기전엔 더하기 1을 해줌 이유는 로그 0 이 되어버리면 값을 정의할수없기때문에

################################### 제출용 제작############################################
results = model.predict(test_file)

sampleSubmission_file['count'] = results

print(sampleSubmission_file[:10])

sampleSubmission_file.to_csv(path + "submit.csv", index = False)
##########################################################################################
#RMSE결과값
'''
loss :  24295.515625
r2스코어 :  0.24918495806383345
RMSE :  154.05051239186346
'''
#RMSLE결과값
'''
loss :  1.5395662784576416
r2스코어 :  0.22796990497314895
RMSE :  1.229973580148046
'''

'''
숙제내용

평균(mean)은 데이터를 모두 더한 후 데이터의 갯수로 나눈 값이다.
중앙값(median)은 전체 데이터 중 가운데에 있는 수이다.
데이터의 수가 짝수인 경우는 가장 가운데에 있는 두 수의 평균이 중앙값이다. 
직원이 100명인 회사에서 직원들 연봉 평균은 5천만원인데 사장의 연봉이 100억인 경우, 
회사 전체의 연봉 평균은 1억 4851만 원이 된다. 이처럼 극단적인 값이 있는 경우 중앙값이 평균값보다 유용하다.
'''