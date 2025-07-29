from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'new_test.csv',)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',index_col=0)
x = train_csv.drop(['count'], axis=1)
# print(x)    #[10886 rows x 10 columns]
y = train_csv['count']
# print(y)    #(10886,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# # exit()
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
srt_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=100, batch_size=32,
                 verbose=2, validation_split=0.2)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)
print('걸린 시간 :', end_time - srt_time, '초')

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + 'submission.csv', index=False)

# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.title('자전거 count')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()

# RMSE : 0.2650455296963583
# R2 : 0.999997805080169