import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

path = "../_data/project/"
df = pd.read_csv(path + '강남역_밥집.csv')

# x_train, x_test, y_train, y_test = train_test_split()
# print(df)

# df = df.loc[df['상권업종대분류명'] == '음식']
# df = df[['상호명', '상권업종중분류명', '상권업종소분류명', '표준산업분류명', '행정동명', '위도', '경도']]
# df = df.loc[(df['행정동명'] == '서초4동') | (df['행정동명'] == '역삼1동')]


# 칼럼명 단순화

df.columns = ['star_point',  # 별점
              'name',  # 가게명
              'dong',  # 위치
              'menu',  # 메뉴
              'view',  # 조회수
              'favorites',  # 즐겨찾기
              'call',  # 전화
              'price'  # 가격
              ]
# print(df) # (217, 8)
# df.info()
df['cate_mix'] = str(df['menu'] + df['name'])
# df['cate_mix'] = df['cate_mix'].str.replace("/", " ")

from sklearn.feature_extraction.text import CountVectorizer  # 피처 벡터화
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도

count_vect_category = CountVectorizer(min_df=0, ngram_range=(1,2))
place_category = count_vect_category.fit_transform(df['cate_mix'])
place_simi_cate = cosine_similarity(place_category, place_category)
place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]

place_simi_co = (
                 + place_simi_cate * 0.3 # 공식 1. 카테고리 유사도
                 + np.repeat([df['view'].values], len(df['view']) , axis=0) * 0.001  # 공식 1. 조회수가 얼마나 많이 나왔는지
                 + np.repeat([df['star_point'].values], len(df['star_point']) , axis=0) * 0.001 # 공식 2. 별점이 얼마나 높은지
                 + np.repeat([df['call'].values], len(df['call']) , axis=0) * 0.001    # 공식 3. 전화가 얼마나 많이 왔는지
                 + np.repeat([df['favorites'].values], len(df['favorites']) , axis=0) * 0.001 # 공식 4. 즐겨찾기가 얼마나 많이 됐는지
                 )



# 아래 place_simi_co_sorted_ind 는 그냥 바로 사용하면 됩니다.
place_simi_co_sorted_ind = place_simi_co.argsort()[:, ::-1] 


# 최종 구현 함수
def find_simi_place(df, sorted_ind, place_name, top_n=10):
    
    place_title = df[df['name'] == place_name]
    place_index = place_title.index.values
    similar_indexes = sorted_ind[place_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)
    return df.iloc[similar_indexes]


# 상도국수를 포함해 5개 업체를 뽑아봅시다.

print(find_simi_place(df, place_simi_co_sorted_ind, '마녀주방', 5))
