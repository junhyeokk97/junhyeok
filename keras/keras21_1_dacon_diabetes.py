import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

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

print(x)
print(y)
print(x.shape)  # (652, 8)
print(y.shape)  # (652,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=57)

print(x_train.shape, x_test.shape)  # (586, 8) (66, 8)
print(y_train.shape, y_test.shape)  # (586,) (66,)

model = Sequential()
model.add(Dense(25, input_dim=8, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   patience = 10, restore_best_weights= True,)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, 
                 callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse: ', rmse)
print('loss: ', loss)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score: ', acc_score)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['Outcome'] = y_submit

# submission_csv.to_csv( path + 'submission_0527_1538.csv', index=False)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('aa loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()


# patience = 300       epochs=10000     41
# RMSE :  0.41504568420819854

# acc_score:  0.7424242424242424


# patience = 300      epochs=10000        random_state=56
# RMSE :  0.35328648410162566
# loss:  0.3985725939273834
# acc_score:  0.8787878787878788