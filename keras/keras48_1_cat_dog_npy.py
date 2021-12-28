import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
# np.save('./_save_npy/keras48_1_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras48_1_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras48_1_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras48_1_test_y.npy', arr=xy_test[0][1])
x_train = np.load('./_save_npy/keras48_1_train_x.npy')
y_train = np.load('./_save_npy/keras48_1_train_y.npy')
x_test = np.load('./_save_npy/keras48_1_test_x.npy')
y_test = np.load('./_save_npy/keras48_1_test_y.npy')

print(x_train.shape)# (8005, 50, 50, 3)
print(x_test.shape)# (2023, 50, 50, 3)
print(y_train.shape) # (8005,)
print(y_test.shape) # (2023,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(50, 50, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=1, validation_split=0.2, callbacks=[es])


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1]*100)
print("%s: %.2f%%" %(model.metrics_names[1], loss[1]*100))
# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/cat_dog/'
sample_image = sample_directory + "euntak.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# print("-- Evaluate --")
# scores = model.evaluate_generator(xy_test, steps=5)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = keras_image.load_img(str(sample_image), target_size=(50, 50))
x = keras_image.img_to_array(image_) # 이미지를 리스트형태의 행열로 받겠다.
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
# y_test.reset()
# print(y_test.class_indices)
# {'cat': 0, 'dog': 1}
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 cat 입니다")
elif(classes[0][0]>0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 dog 입니다")
else:
    print("ERROR")
'''
Epoch 00038: early stopping
64/64 [==============================] - 0s 3ms/step - loss: 0.6734 - accuracy: 0.5774
loss :  0.6733505129814148
-- Predict --
[[0.54486245]]
당신은 54.49 % 확률로 dog 입니다
'''