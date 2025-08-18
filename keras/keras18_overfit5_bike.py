import numpy as np   
import pandas as pd 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 시스템에 따라 폰트 지정
if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    font_name = 'AppleGothic'
else:  # Linux (예: Colab)
    font_name = 'NanumGothic'

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv', index_col=0)
test_csv =  pd.read_csv(path + '/test.csv', index_col=0)
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=4245)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=8))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.1)

'''
# 4. 평가, 예측
results= model.predict(x_test)
rmse = RMSE(results, y_test)
r2 = r2_score(results, y_test)
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")
'''

# 5. 시각화
losses = history.history['loss']
val_losses = history.history['val_loss']
plt.figure(figsize=(10, 6)) # (width, height)
plt.plot(losses, color='red', label='loss')
plt.plot(val_losses, color='blue', label='val_loss')
plt.title('kaggle Loss와 Val Loss 변화')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.grid() # 격자 표시
plt.legend(['Loss', 'Val Loss'], loc='upper right')
plt.show()