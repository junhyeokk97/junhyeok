from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
num = [154, 331, 486, 713, 784]
# 0.95, 0.99 0.999 0.1

import random
seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/따릉이/'     # 문자+문자 = 문자를 이어서 사용 ( ab + cd = abcd )
train_csv = pd.read_csv(path + 'train.csv', index_col=0)  #index_col=0 >> 1번째 열은 인덱스다.
print(train_csv)        # (1459, 11) [1459 rows x 11 columns] 
                        #  >> (1459, 10) [1459 rows x 10 columns] ID는 index이기 때문에 포함하지 않는다   # X와 y 분리
                                                                                                        #   X = train_csv.drop('count', axis=1)
                                                                                                         #   y = train_csv['count']

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # >> count 값
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)   # [715 rows x 1 columns]  >> 측정 완료 된 값.

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns) # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                         # 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                         # 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         # dtype='object')
print(train_csv.info())

# print(train_csv.describe())

# 1,2는 train
############## 결측치 처리 1. 삭제 ################
# print(train_csv.isnull().sum())   # 결측치의 개수 출력
# print(train_csv.isna().sum())

# train_csv = train_csv.dropna()  # train_csv에 있는 결측치를 삭제하고 다시 그 값을 옮긴다.
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)        # [1328 rows x 10 columns] >> 결측치를 뺀 값


############## 결측치 처리 2. 평균값 넣기 ################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())


############## test ################
print(test_csv.info())   # test_csv 결측치는 절대 dropXXX
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


x = train_csv.drop(['count'], axis=1)   # 행 또는 열 삭제  /  count라는 axis=1 삭제, 행은 axis=0, 열은 axis=1
print(x)    # [1459 rows x 9 columns] >> 결측치를 뺀다면 [1328 x 9]

y = train_csv['count']   # count 컬럼만 빼서 y에 넣겠다.
print(y.shape)      # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)


scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

for i in num: 
    pca = PCA(n_components=i)
    x_train = num.fit_transform(x_train)
    x_test = num.fit_transform(x_test)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(f"95% 설명하려면 필요한 주성분 수: {d}")
    d1 = np.argmax(cumsum >= 0.99) + 1
    print(f"95% 설명하려면 필요한 주성분 수: {d1}")
    d2 = np.argmax(cumsum >= 0.999) + 1
    print(f"95% 설명하려면 필요한 주성분 수: {d2}")
    d3 = np.argmax(cumsum >= 1.0) + 1
    print(f"95% 설명하려면 필요한 주성분 수: {d3}")

    model = Sequential()
    model.add(Dense(100, input_dim=8))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, batch_size=200, verbose=2, random_state=50, epochs=50)
    
    results = model.score(x_test, y_test)

    print(x_train.shape, '의 score: ', results)

# 5개 모델 만들기.
#input_shape=
# (70000,154)
# (70000,331)
# (70000,486)
# (70000,713)

