import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
size = 5    #x 4개, y 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)
ccc = split_x(x_predict, size)
x = bbb[:, :4]
y = bbb[:, 4]
x_test = ccc[:, :4]
y_test = ccc[:, 4]
print(x, y)

print(x.shape, y.shape)# (96, 4) (96,)
print(x_test.shape, y_test.shape)



# #2. 모델구성
model = Sequential()
model.add(Dense(60, input_shape = (4,))) # (N, 3, 1) -> N, 10
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))

# model.summary()

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y ,epochs=100)


#4. 평가, 예측
model.evaluate(x, y)
result = model.predict(x_test)
print(result)