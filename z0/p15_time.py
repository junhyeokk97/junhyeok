from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

# 1. 데이터 준비
x_train = np.array(range(100))
y_train = np.array(range(100))

# 모델 생성 함수
def build_model():
    model = Sequential()
    model.add(Dense(3, input_dim=1))
    model.add(Dense(2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# ========== #1. verbose 에 따른 시간 측정 ==========

print("============== [ #1. verbose 에 따른 시간 비교 - batch_size=32 ] ==============")

# model1: verbose=0
model1 = build_model()
start_time = time.time()
model1.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0)
end_time = time.time()
# print(f"model1 (verbose=0) → 걸린 시간: {end - start:.2f}초")
print("걸린시간 : ", end_time - start_time, '초')

# # model2: verbose=1
# model2 = build_model()
# start = time.time()
# model2.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1)
# end = time.time()
# print(f"model2 (verbose=1) → 걸린 시간: {end - start:.2f}초")

# # model3: verbose=2from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

# 데이터
x_train = np.array(range(100))
y_train = np.array(range(100))

# 모델 함수 정의
def build_model():
    model = Sequential()
    model.add(Dense(3, input_dim=1))
    model.add(Dense(2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model




# print("걸린시간 : ", end_time - start_time, '초')

# model1: verbose=0
model1 = build_model()
start_time = time.time()
model1.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0)
end_time = time.time()
print(f"model1 (verbose=0) → 걸린 시간: {end_time - start_time:.2f}초")

# model2: verbose=1
model2 = build_model()
start_time = time.time()
model2.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1)
end_time = time.time()
print(f"model2 (verbose=1) → 걸린 시간: {end_time - start_time:.2f}초")

# model3: verbose=2
model3 = build_model()
start_time = time.time()
model3.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2)
end_time = time.time()
print(f"model3 (verbose=2) → 걸린 시간: {end_time - start_time:.2f}초")

# model4: verbose=3
model4 = build_model()
start_time = time.time()
model4.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=3)
end_time = time.time()
print(f"model4 (verbose=3) → 걸린 시간: {end_time - start_time:.2f}초")
# model3 = build_model()
# start = time.time()
# model3.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2)
# end = time.time()
# print(f"model3 (verbose=2) → 걸린 시간: {end - start:.2f}초")

# # model4: verbose=3
# model4 = build_model()
# start = time.time()
# model4.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=3)
# end = time.time()
# print(f"model4 (verbose=3) → 걸린 시간: {end - start:.2f}초")

# # ========== #2. batch_size 에 따른 시간 측정 ==========

# print("\n============== [ #2. batch_size 에 따른 시간 비교 - verbose=1 ] ==============")

# # model1: batch_size=1
# model1 = build_model()
# start = time.time()
# model1.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1)
# end = time.time()
# print(f"model1 (batch_size=1) → 걸린 시간: {end - start:.2f}초")

# # model2: batch_size=32
# model2 = build_model()
# start = time.time()
# model2.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1)
# end = time.time()
# print(f"model2 (batch_size=32) → 걸린 시간: {end - start:.2f}초")

# # model3: batch_size=64
# model3 = build_model()
# start = time.time()
# model3.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=1)
# end = time.time()
# print(f"model3 (batch_size=64) → 걸린 시간: {end - start:.2f}초")

# # model4: batch_size=128
# model4 = build_model()
# start = time.time()
# model4.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1)
# end = time.time()
# print(f"model4 (batch_size=128) → 걸린 시간: {end - start:.2f}초")
