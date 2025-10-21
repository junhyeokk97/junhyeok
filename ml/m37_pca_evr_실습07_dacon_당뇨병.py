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
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape)  # (652, 9)
print(test_csv.shape)   # (116, 8)
print(submission_csv.shape) # (116, 2)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())
print(train_csv.isnull().sum())
print(train_csv.isna().sum())

x = train_csv.replace(0, np.nan)
x = train_csv.fillna(train_csv.mean())

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50,)
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
    model.add(Dense(1, activation= 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, batch_size=200, verbose=2, random_state=50, epochs=50)
    
    results = model.score(x_test, y_test)

    print(x_train.shape, '의 score: ', results)

# 5개 모델 만들기.
#input_shape=
# (70000,154)
# (70000,331)
# (70000,486)
# (70000,713)

