import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt

path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()       # train_csv에 결측치 데이터를 삭제 처리해라. 결측치 삭제하고 남은 데이터를 반환해서 덮어쓴다

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count'] 

y_log = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 3. 모델 학습
model = Ridge()
model.fit(x_train, y_train)

# 4. 예측 및 역변환
y_pred = model.predict(x_test)

# 5. 평가 (원래 스케일 기준)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)
log_data = np.log1p(y_test)   # np.expmlp(data)
plt.subplot(1,2,1)
plt.hist(y_log, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()

# RMSE: 53.7941
# r2:  0.6030700942121431