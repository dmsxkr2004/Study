from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.' # 띄어쓰기 단위 어절 글단위 음절
text2 = '나는 매우 매우 잘생긴 지구용사 태권브이'
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)

# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
# {'매우': 1, '나는': 2, '진짜': 3, '마구': 4, '맛있는': 5, '밥을': 6, '먹었다': 7, '잘생긴': 8, '지구용사': 9, '태권브이': 10}
x = token.texts_to_sequences([text1, text2])

print(x)
#[[3, 1, 4, 5, 6, 1, 2, 2, 7]]
#[[2, 3, 1, 5, 6, 3, 4, 4, 7], [2, 1, 1, 8, 9, 10]]
x = x[0] + x[1]
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size) 
# 7 , 10
x = to_categorical(x)
# x2 = to_categorical(x[1])
print(x)

'''
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
'''
print(x.shape) # (1, 9, 8)
