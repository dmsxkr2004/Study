import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
# model = VGG16()
model = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
####################### include_top = True ###########################
# 1. FC layer 원래꺼 그대로 쓴다.
# 2. input_shape = (224, 224, 3) 고정 - 못바꾼다.
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#  ................................................................
#  fc1 (Dense)                 (None, 4096)              102764544

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
#######################################################################

####################### include_top = False ###########################
# 1. FC layer 원래꺼 삭제!! - > 앞으로 난 커스터마이징을 할거야!!!
# 2. input_shape = (32, 32, 3) 바꿀수 있다. - 커스터마이징을 할거야!!
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#  ................................................................
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
########################################################################

