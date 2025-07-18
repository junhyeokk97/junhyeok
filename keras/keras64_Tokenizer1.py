import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text = '오늘도 영어를 디게 디게 못 하는 이샥이는 재미 없는 개그를 마구 마구 하면서 딴 짓을 한다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'법률이': 1, '정하는': 2, '형사피의자': 3, '또는': 4, '형사피고인으로서': 5, '구금되었던': 6, '자가': 7,
# '불기소처분을': 8, '받거나': 9, '무죄판결을': 10, '받은': 11, '때에는': 12, '바에': 13, '의하여': 14,
# '국가에': 15, '정당한': 16, '보상을': 17, '청구할': 18, '수': 19, '있다': 20}

print(token.word_counts)
# OrderedDict([('형사피의자', 1), ('또는', 1), ('형사피고인으로서', 1), ('구금되었던', 1), ('자가', 1), ('법률이', 2), ('정하는', 2),
# ('불기소처분을', 1), ('받거나', 1), ('무죄판결을', 1), ('받은', 1), ('때에는', 1),
# ('바에', 1), ('의하여', 1), ('국가에', 1), ('정당한', 1), ('보상을', 1), ('청구할', 1), ('수', 1), ('있다', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[3, 4, 5, 6, 7, 1, 2, 8, 9, 10, 11, 12, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20]]

###### OneHot Encoder 3가지 ######
#1. pandas
x1 = pd.get_dummies(np.array(x).reshape(-1, ))
print(x1)
print(x1.shape) # (16, 14)

#2. sklearn
x2 = np.reshape(x, (-1 ,1))
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(x)
print(x2)
print(x2.shape) # (1, 16)



# #3. keras
x3 = to_categorical(x)
print(x3)
x3 = x3[:,:,1:]
x3 = x3.reshape(16, 14)
print(x3.shape) # (16, 14)