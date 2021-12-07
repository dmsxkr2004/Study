from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape = (10, 10, 1))) # 9,9,10
#input_shape = (a, b, c)ㅡ> a - kernel_size + 1
model.add(Conv2D(5, (3,3), activation = 'relu')) # 7,7,5
model.add(Conv2D(7, (2,2), activation = 'relu')) # 6,6,7
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation = 'softmax'))
model.summary()

# 반장, 이한, 예람, 명재, 모나리자 ->
# LabelEncoder
# 0, 1, 2, 3, 4     ->(5, )             -> (5,1)
                 #[0, 1, 2, 3, 4]       [[0], [1], [2], [3], [4]]

'''
첫 번째 conv2d 에서는 (10, 10, 1) 이미지에 (2, 2) 필터를 10개 사용하였습니다.
이 때, 어떻게 파라미터의 갯수가50개가 나올 수 있을까요?
정확한 내용은 본 블로그의 dl-concept 에서 cnn 블로그 내용을 확인해 보시면 도움이 되겠습니다.
먼저 (2, 2) 필터 한개에는 2 x 2 = 4개의 파라미터가 있습니다.
그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해집니다.
그리고 Conv2D(10, ...) 에서의 10는 10개의 필터를 적용하여 다음 층에서는 채널이 총 10개가 되도록 만든다는 뜻입니다.
여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 10개가 추가로 더해지게 됩니다.

파라미터 구하는법
정리하면, 2 x 2(필터 크기) x 1 (#입력 채널(RGB)) x 10(#출력 채널) + 10(출력 채널 bias) = 50이 됩니다.
(10 x 5)노드 수  x (3 x 3)(필터 크기) x 1 (#입력 채널(RGB)) X  5(#출력 채널) + 5(출력 채널 bias) = 455
(5 x 7)노드 수 x (2 x 2)(필터 크기) x 1 (#입력 채널 (RGB)) x 7(#출력 채널) + 7(출력 채널 bias) = 147
끝 = 파라미터 총 합 = 652
'''

'''
용어 정리
Convolution(합성곱)
채널(Channel)
필터(Filter)
커널(Kernel)
스트라이드(Strid)
패딩(Padding)
피처 맵(Feature Map)
액티베이션 맵(Activation Map)
풀링(Pooling) 레이어
'''