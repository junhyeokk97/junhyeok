import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.optimizers import Adam, Adagrad, SGD, RMSprop # 옵티마이저 클래스 임포트

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

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
        model.add(Dense(16, input_dim=8, activation='relu'))
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