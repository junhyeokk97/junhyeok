# train.csv와 new_test.csv로 count 예측

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
path = '_data/kaggle/bike/'
train_df = pd.read_csv(path + 'train.csv', index_col=0)
new_test_df = pd.read_csv(path + 'new_test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

X = train_df.drop(columns=['count',], axis=1)
y = train_df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# 2. Model Configuration
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=10))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. Compile, Training
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=16)

# 4. Evaluation, Predict
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

submission_csv['count'] = model.predict(new_test_df)
submission_csv.to_csv(path + f'submission_rmse_{rmse:.2f}.csv')