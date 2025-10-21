import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text1 = '오늘도 영어를 디게 디게 못 하는 이샥이는 재미 없는 개그를 마구 마구 하면서 딴 짓을 한다.'

text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. 상진이는 마구 마구 딴 짓을 한다. \
        재현이는 재미없는 딴 짓을 한다.'
        
        
token = Tokenizer()
token.fit_on_texts([text1, text2])
print(token.word_index)
# {'디게': 1, '마구': 2, '딴': 3, '짓을': 4, '한다': 5, '오늘도': 6, '영어를': 7, '못': 8, '하는': 9, '이샥이는': 10,
# '재미': 11, '없는': 12, '개그를': 13, '하면서': 14, '박석사가': 15, '자아를': 16, '찾아냈다': 17, '상진이는': 18,
# '재현이는': 19, '재미없는': 20}

print(token.word_counts)
# OrderedDict([('오늘도', 2), ('영어를', 1), ('디게', 4), ('못', 1), ('하는', 1), ('이샥이는', 1), ('재미', 1), ('없는', 1),
#              ('개그를', 1), ('마구', 4), ('하면서', 1), ('딴', 3), ('짓을', 3), ('한다', 3), ('박석사가', 1), ('자아를', 1),
#              ('찾아냈다', 1), ('상진이는', 1), ('재현이는', 1), ('재미없는', 1)])

x = token.texts_to_sequences([text1, text2])
print(x)
# [[6, 7, 1, 1, 8, 9, 10, 11, 12, 13, 2, 2, 14, 3, 4, 5],
# [6, 15, 16, 1, 1, 17, 18, 2, 2, 3, 4, 5, 19, 20, 3, 4, 5]]


x = np.array(x)
print(x)
# [list([6, 7, 1, 1, 8, 9, 10, 11, 12, 13, 2, 2, 14, 3, 4, 5])
#  list([6, 15, 16, 1, 1, 17, 18, 2, 2, 3, 4, 5, 19, 20, 3, 4, 5])]

x = np.concatenate(x)
print(x)
# [ 6  7  1  1  8  9 10 11 12 13  2  2 14  3  4  5  6 15 16  1  1 17 18  2
#   2  3  4  5 19 20  3  4  5]
# 서로 수치가 다른 리스트를 numpy 배열로 변환하기 위해서는 concat을 사용해 하나로 묶어서 처리도 가능.

###### OneHot Encoder 3가지 ######
# #1. pandas
# x1 = pd.get_dummies(np.array(x).reshape(-1, ))
# print(x1)
# print(x1.shape) # (33, 20)

#2. sklearn
x2 = np.reshape(x, (-1, ))
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(x)
print(x2)
print(x2.shape) # (, )



# # #3. keras
# x3 = to_categorical(x)
# print(x3)
# x3 = x3[:,:,1:]
# # x3 = x3.reshape(18,13)
# print(x3.shape)