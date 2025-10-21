from sklearn.datasets import load_diabetes
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt


# 1. 데이터 로드 및 변환
x, y = load_diabetes(return_X_y=True)
y_log = np.log1p(y)

# 2. train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.1, random_state=42)

# 3. 모델 학습
model = Ridge()
model.fit(x_train, y_train)

# 4. 예측 및 역변환
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)  # ← 예측된 log 스케일을 복원

# 5. 평가 (원래 스케일 기준)
y_test = np.expm1(y_test)
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


# RMSE: 0.9793
# r2:  0.26821941407808403