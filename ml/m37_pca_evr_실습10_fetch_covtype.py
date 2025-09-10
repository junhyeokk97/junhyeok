from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
num = [154, 331, 486, 713, 784]
# 0.95, 0.99 0.999 0.1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]),
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

y = pd.get_dummies(y)
print(y)
print(y.shape)  # (581012, 7)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50,
                                                    stratify=y)


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
    model.add(Dense(10, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, batch_size=200, verbose=2, random_state=50, epochs=50)
    
    results = model.score(x_test, y_test)

    print(x_train.shape, '의 score: ', results)

# 5개 모델 만들기.
#input_shape=
# (70000,154)
# (70000,331)
# (70000,486)
# (70000,713)

