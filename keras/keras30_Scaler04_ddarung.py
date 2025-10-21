 #https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def apply_and_train_with_scalers(X, y, test_size=0.2, random_state=42):
    """
    MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler를 각각 적용하여 데이터를 스케일링하고
    각 스케일링된 데이터로 모델을 학습합니다.

    Args:
        X (pd.DataFrame or np.array): 특성(피처) 데이터.
        y (pd.Series or np.array): 타겟 데이터.
        test_size (float): 학습/테스트 데이터셋 분할 비율. 기본값 0.2.
        random_state (int): 데이터 분할 시 랜덤 시드. 기본값 42.
        model_class (class): 학습할 모델의 클래스 (예: LinearRegression, LogisticRegression 등).
                             기본값은 LinearRegression입니다.

    Returns:
        dict: 각 스케일러별 스케일링된 데이터 (DataFrame)와 학습된 모델을 포함하는 딕셔너리.
              예: {'MinMaxScaler': {'scaled_data': DataFrame, 'model': trained_model}, ...}
    """

    # 원본 데이터의 컬럼 이름을 유지하기 위해 DataFrame으로 변환 (넘파이 배열이 들어올 경우)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    results = {}
    scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler()
    }

    # 학습/테스트 데이터 분할 (스케일링 전에 분할하여 데이터 누수 방지)
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    for scaler_name, scaler in scalers.items():
        print(f"Applying {scaler_name}...")

        # 스케일러 학습 및 변환
        X_train_scaled = scaler.fit_transform(X_train_orig)
        X_test_scaled = scaler.transform(X_test_orig)

        # 스케일링된 데이터를 DataFrame으로 변환 (컬럼명 유지)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_orig.columns, index=X_train_orig.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_orig.columns, index=X_test_orig.index)

        # 전체 스케일링된 데이터 (참고용, 실제 모델 학습은 train/test로 진행)
        X_scaled_full = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


        # 모델 학습
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        
        model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_split=0.1, callbacks=[es])

        results[scaler_name] = {
            'scaler': scaler,  # 사용된 스케일러 객체
            'X_train_scaled': X_train_scaled_df,
            'X_test_scaled': X_test_scaled_df,
            'y_train': y_train, # 스케일링되지 않은 y_train
            'y_test': y_test,   # 스케일링되지 않은 y_test
            'X_scaled_full': X_scaled_full, # 전체 데이터셋 스케일링된 버전 (참고용)
            'model': model
        }
        print(f"{scaler_name} applied and model trained.")

    return results

# 사용 예시:
if __name__ == "__main__":

    path = './_data/dacon/따릉이/'         
    train_df = pd.read_csv(path +'/train.csv', index_col = 0)

    train_df = train_df.fillna(train_df.mean())

    X = train_df.drop(['count'], axis=1)
    y = train_df['count']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=18,
        shuffle=True
    )

    print("===== Starting scaling and model training =====")
    scaled_results = apply_and_train_with_scalers(X, y)
    print("===== All operations completed =====")

    # 결과 확인
    for scaler_name, data in scaled_results.items():
        print(f"\n--- Results for {scaler_name} ---")
        print(f"X_train_scaled shape: {data['X_train_scaled'].shape}")
        print(f"X_test_scaled shape: {data['X_test_scaled'].shape}")

        # 모델 성능 평가
        y_pred = data['model'].predict(data['X_test_scaled'])
        mse = RMSE(data['y_test'], y_pred)
        print(f"Model ({scaler_name}) Mean Squared Error on test set: {mse:.2f}")

'''
--- Results for MinMaxScaler ---
X_train_scaled shape: (1167, 9)
X_test_scaled shape: (292, 9)
Model (MinMaxScaler) Root Mean Squared Error on test set: 47.71

--- Results for StandardScaler ---
X_train_scaled shape: (1167, 9)
X_test_scaled shape: (292, 9)
Model (StandardScaler) Root Mean Squared Error on test set: 45.67

--- Results for MaxAbsScaler ---
X_train_scaled shape: (1167, 9)
X_test_scaled shape: (292, 9)
Model (MaxAbsScaler) Root Mean Squared Error on test set: 50.40

--- Results for RobustScaler ---
X_train_scaled shape: (1167, 9)
X_test_scaled shape: (292, 9)
Model (RobustScaler) Root Mean Squared Error on test set: 45.46
'''