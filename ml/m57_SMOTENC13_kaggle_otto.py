# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

# 1. 라이브러리 import
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    roc_auc_score, log_loss
)
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTENC

# 2. 경로 설정 및 데이터 로드
path = './_data/kaggle/otto/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sampleSubmission.csv')

# 3. 데이터 분리
X = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

# 4. 라벨 인코딩 및 스케일링
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0 ~ 8 정수 인코딩
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

smotenc = SMOTENC(random_state=50,
                  categorical_features=[2,3,6,7,8])

# 5. 학습/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6. One-hot 인코딩
y_train_ohe = to_categorical(y_train, num_classes=9)
y_val_ohe = to_categorical(y_val, num_classes=9)

# 스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# # loss: 0.7634
# # acc_score: 0.7253
# # AUC: 0.963
# # Mean Squared Error: 2.8542
# # R2 score: 0.5472
# # Log Loss: 0.7634
# # Accuracy Score: 0.7253
# # AUC (macro): 0.9452
scaler = StandardScaler()
# # loss: 0.4725
# # acc_score: 0.8166
# # AUC: 0.9839
# # Mean Squared Error: 1.6618
# # R2 score: 0.7364
# # Log Loss: 0.4725
# # Accuracy Score: 0.8166
# # AUC (macro): 0.9766
# scaler = MaxAbsScaler()
# # loss: 0.7567
# # acc_score: 0.726
# # AUC: 0.9634
# # Mean Squared Error: 2.9389
# # R2 score: 0.5338
# # Log Loss: 0.7567
# # Accuracy Score: 0.726
# # AUC (macro): 0.9453
# scaler = RobustScaler()
# # loss: 0.4756
# # acc_score: 0.8151
# # AUC: 0.9839
# # Mean Squared Error: 1.7298
# # R2 score: 0.7256
# # Log Loss: 0.4756
# # Accuracy Score: 0.8151
# # AUC (macro): 0.9762
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_val)

exit()
# 7. 모델 구성
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(9, activation='softmax')  # 9개 클래스
])

# 함수형 모델

input1 = Input(shape=(93,))
dense1 = Dense(256)(input1)  #summary 했을때 name으로 layer를 지정한 것으로 지을 수 있다
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(128)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(64)(drop2)
drop3 = Dropout(0.2)(dense3)
output1 = Dense(9)(drop3)


model2 = Model(inputs=input1, outputs=output1)
model2.summary()

# 8. 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 9. 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

start = time.time()

# 10. 모델 학습
hist = model.fit(
    X_train, y_train_ohe,
    validation_data=(X_val, y_val_ohe),
    epochs=100,
    batch_size=512,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

end = time.time()

# 모델 요약
model.summary()

# 결과 출력
print("============= hist =============")
print(hist)
print("========= hist.history =========")
print(hist.history)
print("========== val_loss ===========")
print(hist.history['val_loss'])

###그래프를 그려보자
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 주로 사용
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

plt.figure(figsize=(9, 6))  # 도표 사이즈 지정

plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('otto Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시
plt.legend(loc='upper right')  # 우측 상단 범례

plt.show()


# 11. 평가 지표 계산
results = model.evaluate(X_val, y_val_ohe, verbose=0)
print("loss:", round(results[0], 4))
print("acc_score:", round(results[1], 4))
print("AUC:", round(results[2], 4))

y_val_pred_proba = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
logloss = log_loss(y_val_ohe, y_val_pred_proba)
acc = accuracy_score(y_val, y_val_pred)
auc_macro = roc_auc_score(y_val_ohe, y_val_pred_proba, multi_class='ovr')

print("Mean Squared Error:", round(mse, 4))
print("R2 score:", round(r2, 4))
print("Log Loss:", round(logloss, 4))
print("Accuracy Score:", round(acc, 4))
print("AUC (macro):", round(auc_macro, 4))

# 12. 테스트 예측 및 제출 파일 생성
test_pred_proba = model.predict(X_test_scaled)
submission[le.classes_] = test_pred_proba
submission.to_csv('otto_submission_sgd(2)', index=False)
print("제출 파일 생성 완료: otto_submission_sgd(1).csv")

submission_path = path + 'sampleSubmission.csv'
if not os.path.exists(submission_path):
    print("❌ sample_submission.csv 파일이 존재하지 않습니다.")
else:
    submission = pd.read_csv(submission_path)
    print("✅ sample_submission.csv 로드 성공!")

# loss: 0.4789
# acc_score: 0.8119
# AUC: 0.9838
# Mean Squared Error: 1.7037
# R2 score: 0.7297
# Log Loss: 0.4789
# Accuracy Score: 0.8119
# AUC (macro): 0.9756

