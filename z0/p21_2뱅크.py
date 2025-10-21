import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 불러오기
path = 'C:/Study25/_data/kaggle/Bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 2. 특성 선택
X = train_csv.drop(columns=['CustomerId', 'Surname', 'Exited'])  # 입력
y = train_csv['Exited']                                          # 타깃
X_test = test_csv.drop(columns=['CustomerId', 'Surname'])        # 테스트 입력

# 3. Label Encoding
for col in ['Geography', 'Gender']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# 4. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. StratifiedKFold 교차검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds = np.zeros((X_test.shape[0],))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 모델 구성
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=100,
              batch_size=512,
              callbacks=[es],
              verbose=0)

    # 평가
    val_pred = (model.predict(X_val) > 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    print(f'Fold {fold+1} Accuracy: {acc:.4f}')
    val_scores.append(acc)

    # 테스트 예측 누적
    test_preds += model.predict(X_test_scaled).reshape(-1)

# 평균 정확도
print(f'\nMean CV Accuracy: {np.mean(val_scores):.4f}')

# 6. 결과 저장 (0.5 기준 이진화)
submission_csv['Exited'] = (test_preds / skf.n_splits > 0.5).astype(int)
submission_csv.to_csv(path + 'submission_ann.csv')
print("제출 파일 저장 완료 ✅")
