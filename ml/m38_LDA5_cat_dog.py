from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler

#1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
x = np.load(np_path + "keras44_02_x_train.npy")
y = np.load(np_path + "keras44_02_y_train.npy")
test = np.load(np_path + "keras44_02_test.npy")

x = x.reshape(x.shape[0], 100*100)
y = y.reshape(-1)
test = test.reshape(test.shape[0], -1)
print(test.shape)
print(x.shape, y.shape) # (30000, 100, 100, 1) (30000,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)


### scaler는 pca 전에 하는 게 좋음.
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)
print(x_test.shape)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
# LDA의 n_component는 y label 갯수 -1 이하로 만들 수 있다.
lda = LinearDiscriminantAnalysis(n_components=1)    # 50을 제외하고 세밀하게 나눈다?
train_lda = lda.fit_transform(x_train, y_train) # LDA는 y값도 같이 transform
test_lda = lda.transform(x_test)   # y_test에 transform한 값들이 들어가기 때문에 y_test는 lda를 거치지 않아도 됨?
# print(x)
# print(x.shape)

model = RandomForestClassifier(random_state=50)

model.fit(train_lda, y_train)

results = model.score(test_lda, y_test)

print('score: ', results)

# score:  0.5072