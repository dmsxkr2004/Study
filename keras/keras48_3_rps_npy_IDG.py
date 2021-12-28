import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy, categorical_crossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
# 1. 데이터

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip= True,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split = 0.3
    )

# test_datagen = ImageDataGenerator(
#     rescale = 1./255
# )
# D:\_data\image\rps\rps
xy_train = train_datagen.flow_from_directory(         
    '../_data/image/rps/rps',
    target_size = (50,50),
    batch_size = 1770,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True
    )
'''
Found 1764 images belonging to 3 classes.
'''

validation_datagen = train_datagen.flow_from_directory(
    '../_data/image/rps/rps',
    target_size = (50,50),
    batch_size = 760, 
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True
)
'''
Found 756 images belonging to 3 classes.
'''
print(xy_train[0][0].shape, xy_train[0][1].shape) # (10, 50, 50, 3) (10,)
np.save('./_save_npy/keras48_3_train_x.npy', arr=xy_train[0][0])
np.save('./_save_npy/keras48_3_train_y.npy', arr=xy_train[0][1])
np.save('./_save_npy/keras48_3_test_x.npy', arr=validation_datagen[0][0])
np.save('./_save_npy/keras48_3_test_y.npy', arr=validation_datagen[0][1])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(64,kernel_size = (2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu')) 
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

# model.fit(xy_train[0][0], xy_train[0][1])    # x 와 y   validation_steps = 4, 뜻 알아내기
hist = model.fit_generator(xy_train, epochs = 30, steps_per_epoch = 177, # steps_per_epoch : 에포당 스텝을 몇 번할 것인가?? = 전체 데이터 나누기 배치
                    validation_data = validation_datagen,
                    validation_steps = 4,)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심 때 그래프 그려 보기
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])


# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/rps/'
sample_image = sample_directory + "scissors_et.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플케이스 평가

loss, acc = model.evaluate(validation_datagen)
print("Accuracy : ", str(np.round(acc ,2)*100)+ "%")
#TypeError: 'float' object is not subscriptable

image_ = keras_image.load_img(str(sample_image), target_size=(50, 50))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)#NDIMS

validation_datagen.reset()
print(validation_datagen.class_indices)
# {'paper': 0, 'rock': 1, 'scissors': 2}
if(y_predict==0):
    print(" 보 " )
elif(y_predict==1):
    print(" 바위 ")
elif(y_predict==2):
    print(" 가위 ")
else:
    print("ERROR")
