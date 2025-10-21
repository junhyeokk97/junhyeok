import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'samplesubmission.csv')

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (61878, 93)
print(y.shape)  # (61878,)

x = x.values.reshape(x.shape[0], 31, 3)
y = y.values.reshape(y.shape[0], 1)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
scaler.fit(x_test)
x_test = scaler.fit_transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC
model = LinearSVC(C=0.3)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


model_list= [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

for model_set in model_list:
    model = model_set()
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict)
    f1 = f1_score(y_test, y_predict, average='macro')
    print(f'{model_set.__name__} F1 score: {score:.4f}')