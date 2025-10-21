import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로딩
path = 'c:/Study25/_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 입력/출력 나누기
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]

# 정규화
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42
)

# 모델 구성 (좀 더 깊고 넓게)
model = Sequential([
    Dense(128, input_shape=(8,), activation='relu'),    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),    
    Dense(16, activation='relu'),
    Dense(2, activation='linear')  # casual + registered 예측
])

# 컴파일
model.compile(loss='mse', optimizer='adam')

# EarlyStopping 설정
es = EarlyStopping(
    monitor='val_loss',
    patience=30,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 학습
hist = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 예측 및 평가
loss = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"loss: {loss:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

"""
    191/191 [==============================] - 0s 802us/step - loss: 8985.8799 - val_loss: 8814.7031
Epoch 00171: early stopping
loss: 8862.7578
R2 Score: 0.4285
RMSE: 94.1422
"""