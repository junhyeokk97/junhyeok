import numpy as np
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop # 옵티마이저 클래스 임포트

#1. 데이터
path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

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
        model.add(Dense(16, input_dim=200, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        #3. 컴파일, 훈련
        # 옵티마이저 인스턴스를 생성하여 전달합니다.
        # === 핵심 수정 부분 ===
        optimizer_instance = opt_class(learning_rate=lr_value) # 옵티마이저 클래스를 호출하여 인스턴스 생성
        model.compile(loss='binary_crossentropy', optimizer=optimizer_instance)
        # === 핵심 수정 부분 끝 ===
        
        start_time = time.time()
        # verbose=0으로 설정하여 훈련 과정 출력 생략
        hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                         verbose=0, validation_split=0.1)
        end_time = time.time() # time.time()은 함수 호출이 아니므로 괄호 없음 (이전에도 수정됨)
        
        #4. 평가, 예측
        loss = model.evaluate(x_test, y_test, verbose=0) # verbose=0으로 설정하여 평가 과정 출력 생략
        results = model.predict(x_test, verbose=0) # verbose=0으로 설정하여 예측 과정 출력 생략
        y_pred = (results > 0.5).astype(int)
        y_test = y_test.astype(int)
        acc = accuracy_score(y_test, y_pred)
        training_time = end_time - start_time

        print(f"Optimizer: {opt_class.__name__}, LR: {lr_value}")
        print(f"Loss: {loss:.4f}")
        print(f"R2 Score: {acc:.4f}")
        print(f"Training Time: {training_time:.4f} seconds")
        
        all_results.append({
            'optimizer': opt_class.__name__,
            'learning_rate': lr_value,
            'loss': loss,
            'accuracy_score': acc,
            'training_time': training_time
        })

print("\n=======================================================")
print("All Training Results:")
print("=======================================================")
for res in all_results:
    print(f"Optim: {res['optimizer']},\
            LR: {res['learning_rate']:.4f},\
            Loss: {res['loss']:.4f}, R2: {res['accuracy_score']:.4f},\
            Time: {res['training_time']:.4f}s")
    
# Optim: Adam,            LR: 0.1000,            Loss: 0.3289, R2: 0.8994,            Time: 293.0806s
# Optim: Adam,            LR: 0.0100,            Loss: 0.2602, R2: 0.8937,            Time: 298.1312s
# Optim: Adam,            LR: 0.0010,            Loss: 0.2783, R2: 0.9058,            Time: 291.4806s
# Optim: Adam,            LR: 0.0005,            Loss: 0.2981, R2: 0.9033,            Time: 305.0089s
# Optim: Adam,            LR: 0.0001,            Loss: 0.2666, R2: 0.9029,            Time: 301.2349s
# Optim: Adagrad,            LR: 0.1000,            Loss: 0.3097, R2: 0.8889,            Time: 251.0440s
# Optim: Adagrad,            LR: 0.0100,            Loss: 0.2501, R2: 0.9077,            Time: 254.6363s
# Optim: Adagrad,            LR: 0.0010,            Loss: 0.2467, R2: 0.9083,            Time: 258.9316s
# Optim: Adagrad,            LR: 0.0005,            Loss: 0.2559, R2: 0.9039,            Time: 259.7183s
# Optim: SGD,            LR: 0.1000,            Loss: 0.2890, R2: 0.9084,            Time: 238.1519s
# Optim: SGD,            LR: 0.0100,            Loss: 0.2634, R2: 0.9019,            Time: 235.9311s
# Optim: SGD,            LR: 0.0010,            Loss: 0.2361, R2: 0.9141,            Time: 236.3490s
# Optim: SGD,            LR: 0.0005,            Loss: 0.2367, R2: 0.9135,            Time: 232.1307s
# Optim: SGD,            LR: 0.0001,            Loss: 0.2755, R2: 0.8994,            Time: 236.4846s
# Optim: RMSprop,            LR: 0.1000,            Loss: 0.3271, R2: 0.8994,            Time: 260.6377s
# Optim: RMSprop,            LR: 0.0100,            Loss: 0.2956, R2: 0.9004,            Time: 253.6175s
# Optim: RMSprop,            LR: 0.0010,            Loss: 2.0547, R2: 0.9082,            Time: 257.6038s
# Optim: RMSprop,            LR: 0.0005,            Loss: 0.6628, R2: 0.9006,            Time: 258.8836s
# Optim: RMSprop,            LR: 0.0001,            Loss: 0.3172, R2: 0.9059,            Time: 262.1166s