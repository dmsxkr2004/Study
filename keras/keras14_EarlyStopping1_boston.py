from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import time as time
from tensorflow.python.keras.callbacks import History
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 13))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping patience(기다리는 횟수)
es = EarlyStopping(monitor='val_loss', patience=5, mode = 'min', verbose = 1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
                 validation_split = 0.2 , callbacks = [es])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
print("---------------------------------")
print(hist)
print("---------------------------------")
print(hist.history)
print("---------------------------------")
print(hist.history['loss'])
print("---------------------------------")
print(hist.history['val_loss'])
print("---------------------------------")
'''
plt.figure(figsize = (9, 5))
plt.plot(hist.history['loss'], marker =',',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker =',',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

'''
최저치 갱신후 100번째 때에 콜백하는지
최저치 갱신후 100번째 후에 최저치에 콜백하는지 알아내서 쓰기
1000번 다 돌려보고 수치구하고
1000번 돌리는데 최저치에 멈췃을때 수치비교해보고 쓰기
'''
'''
숙제 내용
-----------------------------------------------------------------------------------------------
patience 값 5 주고 돌렸을때의 결과값
323/323 [==============================] - 0s 576us/step - loss: 63.7636 - val_loss: 65.1947
Epoch 00023: early stopping
걸린시간 :  4.962 초
4/4 [==============================] - 0s 1ms/step - loss: 47.6942
loss:  47.69422912597656
r2스코어 :  0.42937841623708173
-----------------------------------------------------------------------------------------------
patience 값 20 주고 돌렸을때의 결과값 
323/323 [==============================] - 0s 642us/step - loss: 31.8540 - val_loss: 34.9224
Epoch 00102: early stopping
걸린시간 :  21.597 초
4/4 [==============================] - 0s 1ms/step - loss: 20.0073
loss:  20.007320404052734
r2스코어 :  0.7606291706594244
-----------------------------------------------------------------------------------------------
Earlystopping 값 끄고 1000번 돌렸을때의 결과값
323/323 [==============================] - 0s 587us/step - loss: 27.2044 - val_loss: 34.4958
걸린시간 :  189.377 초
4/4 [==============================] - 0s 0s/step - loss: 21.5278
loss:  21.52782440185547
r2스코어 :  0.742437579279855
-----------------------------------------------------------------------------------------------
loss, var_loss값 추출 patience 값 5 주고 돌렸을 경우 
---------------------------------
[470.0278015136719, 118.42193603515625, 89.87291717529297, 87.50634002685547, 76.9632568359375, 
79.0855941772461, 79.3959732055664, 86.59437561035156, 76.7964859008789, 72.72969818115234, 
70.39262390136719, 68.73475646972656, 67.84699249267578, 67.929443359375, 66.32917785644531, 
59.63949966430664, 58.601863861083984, 67.08643341064453, 57.82572555541992, 61.9532470703125, 
59.846004486083984, 65.21489715576172, 54.291542053222656, 57.28762435913086, 54.57377624511719, 
57.841758728027344, 52.316810607910156, 50.94084930419922, 50.701236724853516, 53.08031463623047]   
---------------------------------
[103.23992919921875, 73.0350341796875, 82.51017761230469, 71.8651351928711, 139.51686096191406, 
73.58367919921875, 81.32054901123047, 68.97469329833984, 72.576416015625, 114.07636260986328, 
82.17882537841797, 64.52068328857422, 74.5323715209961, 66.17909240722656, 60.18892288208008, 
56.86946487426758, 85.63029479980469, 91.6958999633789, 57.370826721191406, 78.14178466796875, 
54.64942169189453, 80.20538330078125, 69.42469787597656, 55.90462875366211, 53.08073425292969, 
54.59933853149414, 64.79865264892578, 53.12040710449219, 56.081905364990234, 57.59013366699219]    
---------------------------------

## EarlyStopping 개념정리
조기종료(early stopping)은 Neural Network가 과적합을 회피하도록 만드는 정칙화(regularization) 기법 중 하나이다[1]. 
훈련 데이터와는 별도로 검증 데이터(validation data)를 준비하고, 매 epoch 마다 검증 데이터에 대한 오류(validation loss)를 측정하여 모델의 훈련 종료를 제어한다. 
구체적으로, 과적합이 발생하기 전 까지 training loss와 validaion loss 둘다 감소하지만, 과적합이 일어나면 training loss는 감소하는 반면에 validation loss는 증가한다. 
그래서 early stopping은 validation loss가 증가하는 시점에서 훈련을 멈추도록 조종한다.

validaion loss가 감소하다가 21번째부턴 계속해서 증가한다고 가정해보자. 
patience를 5로 설정하였기 때문에 모델의 훈련은 25번째 epoch에서 종료할 것이다. 
그렇다면 훈련이 종료되었을 때 이 모델의 성능은 20번째와 25번째에서 관측된 성능 중에서 어느 쪽과 일치할까? 
안타깝게도 20번째가 아닌 25번째의 성능을 지니고 있다. 위 예제에서 적용된 early stopping은 훈련을 언제 종료시킬지를 결정할 뿐이고, 
Best 성능을 갖는 모델을 저장하지는 않는다. 따라서 early stopping과 함께 모델을 저장하는 callback 함수를 반드시 활용해야만 한다.

개념과 사용시 주의사항들 정리한 사이트
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221468928164
https://forensics.tistory.com/29


callback 함수를 사용하더라도 마지막 출력값이 적용되어서 히스토리내역을 보면 53까지 내려가는 val_loss 값을 출력한게 아니라
57 번째에 출력이 되어서 최저점의 로스값은 아니라고 볼 수 있다 하지만 근사치의 값은 뽑아낸다
'''
