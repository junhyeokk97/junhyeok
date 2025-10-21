from tensorflow.keras.datasets import mnist
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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

for i in num: 
    pca = PCA(n_components=i)
    x_train = num.fit_transform(x_train)
    x_test = num.fit_transform(x_test)

    model = Sequential()
    model.add(Dense(200, input_dim=784))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, batch_size=200, verbose=2, random_state=50, epochs=50)
    
    results = model.score(x_test, y_test)

    print(x_train.shape, '의 score: ', results)

# 5개 모델 만들기.
#input_shape=
# (70000,154)
# (70000,331)
# (70000,486)
# (70000,713)

