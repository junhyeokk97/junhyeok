import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop # 옵티마이저 클래스 임포트

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

############## 결측치 처리 2. 평균값 넣기 ################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())


############## test ################
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


x = train_csv.drop(['count'], axis=1)


y = train_csv['count']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=748
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 사용할 옵티마이저 클래스와 학습률 리스트
optimizers = [Adam, Adagrad, SGD, RMSprop] # 변수명 'optim' -> 'optimizers'로 변경 권장
learning_rates = [0.1, 0.01, 0.001, 0.0005, 0.0001] # 변수명 'lr' -> 'learning_rates'로 변경 권장

# 결과를 저장할 리스트 (선택 사항)
all_results = []

# 옵티마이저와 학습률을 반복하여 모델 훈련
# 외부 for 루프는 옵티마이저 클래스를, 내부 for 루프는 학습률을 반복합니다.
# === 핵심 수정 부분 ===
for opt_class in optimizers: # 옵티마이저 클래스 (Adam, Adagrad 등)를 직접 가져옴
    for lr_value in learning_rates: # 학습률 값 (0.1, 0.01 등)을 직접 가져옴
    # === 핵심 수정 부분 끝 ===

        print(f"\n=======================================================")
        print(f"Training with Optimizer: {opt_class.__name__}, Learning Rate: {lr_value}")
        print(f"=======================================================")

        #2. 모델구성
        model = Sequential()
        model.add(Dense(16, input_dim=9, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        #3. 컴파일, 훈련
        # 옵티마이저 인스턴스를 생성하여 전달합니다.
        # === 핵심 수정 부분 ===
        optimizer_instance = opt_class(learning_rate=lr_value) # 옵티마이저 클래스를 호출하여 인스턴스 생성
        model.compile(loss='mse', optimizer=optimizer_instance)
        # === 핵심 수정 부분 끝 ===

        start_time = time.time()
        # verbose=0으로 설정하여 훈련 과정 출력 생략
        hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                         verbose=0, validation_split=0.1)
        end_time = time.time() # time.time()은 함수 호출이 아니므로 괄호 없음 (이전에도 수정됨)
        
        #4. 평가, 예측
        loss = model.evaluate(x_test, y_test, verbose=0) # verbose=0으로 설정하여 평가 과정 출력 생략
        results = model.predict(x_test, verbose=0) # verbose=0으로 설정하여 예측 과정 출력 생략
        results = np.nan_to_num(results, nan=0.0, posinf=0.0, neginf=0.0)
        r2 = r2_score(y_test, results)
        
        training_time = end_time - start_time

        print(f"Optimizer: {opt_class.__name__}, LR: {lr_value}")
        print(f"Loss: {loss:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Training Time: {training_time:.4f} seconds")
        
        all_results.append({
            'optimizer': opt_class.__name__,
            'learning_rate': lr_value,
            'loss': loss,
            'r2_score': r2,
            'training_time': training_time
        })

print("\n=======================================================")
print("All Training Results:")
print("=======================================================")
for res in all_results:
    print(f"Optim: {res['optimizer']},\
            LR: {res['learning_rate']:.4f},\
            Loss: {res['loss']:.4f}, R2: {res['r2_score']:.4f},\
            Time: {res['training_time']:.4f}s")
    
# Optim: Adam,            LR: 0.1000,            Loss: 5083.2021, R2: -0.0071,            Time: 4.1910s
# Optim: Adam,            LR: 0.0100,            Loss: 1445.2738, R2: 0.7136,            Time: 4.1294s
# Optim: Adam,            LR: 0.0010,            Loss: 1858.6201, R2: 0.6317,            Time: 4.0511s
# Optim: Adam,            LR: 0.0005,            Loss: 1837.6066, R2: 0.6359,            Time: 4.0654s
# Optim: Adam,            LR: 0.0001,            Loss: 1695.0522, R2: 0.6642,            Time: 4.0240s
# Optim: Adagrad,            LR: 0.1000,            Loss: 2224.0681, R2: 0.5593,            Time: 3.6844s
# Optim: Adagrad,            LR: 0.0100,            Loss: 1748.7816, R2: 0.6535,            Time: 3.7478s
# Optim: Adagrad,            LR: 0.0010,            Loss: 2130.3542, R2: 0.5779,            Time: 3.6851s
# Optim: Adagrad,            LR: 0.0005,            Loss: 5335.6094, R2: -0.0572,            Time: 3.7019s
# Optim: Adagrad,            LR: 0.0001,            Loss: 15439.7881, R2: -2.0591,            Time: 3.6792s
# Optim: SGD,            LR: 0.1000,            Loss: nan, R2: -2.0702,            Time: 3.5254s
# Optim: SGD,            LR: 0.0100,            Loss: nan, R2: -2.0702,            Time: 3.4973s
# Optim: SGD,            LR: 0.0010,            Loss: nan, R2: -2.0702,            Time: 3.5300s
# Optim: SGD,            LR: 0.0005,            Loss: nan, R2: -2.0702,            Time: 3.6101s
# Optim: SGD,            LR: 0.0001,            Loss: 1566.4940, R2: 0.6896,            Time: 3.5593s
# Optim: RMSprop,            LR: 0.1000,            Loss: 2095.8823, R2: 0.5847,            Time: 4.0800s
# Optim: RMSprop,            LR: 0.0100,            Loss: 2643.4141, R2: 0.4763,            Time: 4.0104s
# Optim: RMSprop,            LR: 0.0010,            Loss: 1519.0829, R2: 0.6990,            Time: 4.0796s
# Optim: RMSprop,            LR: 0.0005,            Loss: 1620.8002, R2: 0.6789,            Time: 4.0771s
# Optim: RMSprop,            LR: 0.0001,            Loss: 1830.8346, R2: 0.6373,            Time: 4.0179s