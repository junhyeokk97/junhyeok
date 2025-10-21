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
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# test_csv['Geography'] = le_geo.fit_transform(test_csv['Geography'])
# test_csv['Gender'] = le_gen.fit_transform(test_csv['Gender'])

print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884


train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)
print(train_csv.columns)    # ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                            #    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                            #    'Exited']

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)
print(y.value_counts())
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

