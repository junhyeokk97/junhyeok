import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer

# 1. 데이터 로드
datasets = load_breast_cancer()
# print(datasets.DESCR)   # 데이터 설명: 569개의 샘플, 30개의 피처
# print(datasets.feature_names)  # 피처 이름 출력
print("===========================================================")
print(type(datasets))  # <class 'sklearn.utils._bunch.Bunch'>

# 2. 데이터 분리
x = datasets.data          # (569, 30)
y = datasets.target        # (569, ) - 0 or 1 (이진 분류)

print(x.shape, y.shape)    # (569, 30), (569,)
print(type(x))             # <class 'numpy.ndarray'>
print(type(y))             # <class 'numpy.ndarray'>

print(x)
print(y)

# 0과 1의 개수가 몇개인지 찾아보기
# 1. pandas
# 2. numpy

#numpy
print(np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357]))

#pandas
print(pd.value_counts(y))
# 1    357
# 0    212
print(pd.Series(y).value_counts()) # 위랑 같은 형식
print(pd.DataFrame(y).value_counts())

#데이터

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.1, random_state=1375,
    shuffle=True,
)

print(x_train.shape, y_train.shape) #(398, 30) (398,)
print(x_test.shape, y_test.shape)   #(171, 30) (171,)

#모델 구성
model = Sequential()    
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   
                # 바이너리 로지스틱 회귀에 주로 사용할 **시그모이드** >> **0에서 1사이의 값이된다**
                # 양수부분은 0.5를 기준으로 오른쪽으로 무한대 이동하면 1(에 가장 근접한)
                # 0.5를 기준으로 왼쪽으로 무한대 이동하면 0(에 가장 근접한)
                # linear 상태는 0에서 1사이를 통과하는 선
                # sigmoid 상태는 직선 대신, S자 곡선을 이용하여 분류에 정확도를 향상
                # 출력층에 'relu'를 쓰지 않는 이유: ReLU는 0 이상의 값을 그대로 통과시키므로 확률 표현이 불가능하고, 음수는 0이 되어 정보 손실이 생김.
                # 마지막 activation은 시그모이드라고 외워도 무방하나, 수식구조 개념 이해가 필요
                
#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
# binary_cross_entropy
# ㄴ이진분류. y가 0,1  즉, binary_cross_entropy 쓰였냐 안쓰였냐 이진분류는 무조건 시그모이드.
# 출력층 activation >> sigmoid
# 손실 함수 loss=binary_crossentropy

# Metric	설명
# 'accuracy'	정분류 비율 = 'acc'
# 'Precision'	정밀도 (Positive 예측 중 실제 Positive 비율)
# 'Recall'	재현율 (실제 Positive 중 맞춘 비율)
# 'AUC'	ROC 곡선 아래 면적


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping (
    monitor='val_loss', # 모니터링할 지표를 선택    # https://wikidocs.net/194135
    mode='auto',   # 최대값 max, 알아서 찾아봐:auto
    patience=20,    # 개선이 없다고 판단하기 전에 대기할 에폭 수. 
                    # 즉, 이 값만큼 검증 손실이 개선되지 않으면 학습을 중단
    restore_best_weights=True,  #가장 최소 지점을 세이브 할 거냐
)

hist = model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1,     # # over fit
        validation_split=0.2,
        callbacks=[es])

# 딕셔너리에 밸류를 보자.
# loss 개수 = epochs의 값, val_loss 개수 = epochs의 값
print("================ hist ================")                 
print(hist)
print("================ hist.history ================")
print(hist.history)
print("================ val_loss ================")
print(hist.history['loss'])

###그래프를 그려보자
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 주로 사용
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

plt.figure(figsize=(9, 6))  # 도표 사이즈 지정

plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('텐서 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시
plt.legend(loc='upper right')  # 우측 상단 범례

plt.show()

start_time = time.time() 
# 결과, 예측
results = model.evaluate(x_test, y_test)

# loss: 0.3623 - accuracy: 0.8596 print로 loss와 accuracy 나온것을 확인
# - val_loss: 0.14799687266349792
print(results)
print("loss: ", results[0])
print("acc: ", round(results[1], 4))
# - loss: 0.3273 - accuracy: 0.8947
# [0.32727330923080444, 0.8947368264198303]
# loss:  0.32727330923080444
# acc:  0.8947




# from sklearn.metrics import accuracy_score 
# accuracy_score = accuracy_score(y_test, y_predict)  # ❌ 여기서 함수 이름을 덮어씀
# print(y_predict[:10])
# y_predict = np.round(y_predict)  # ❗ 이미 accuracy_score 계산 후 반올림. 순서도 이상함
# print('acc_score:', accuracy_score)

from sklearn.metrics import accuracy_score
import time

start_time = time.time()

# 예측
y_predict = model.predict(x_test)

# 확률값이므로 0.5 기준으로 반올림
y_predict = np.round(y_predict)

# 정확도 계산
acc = accuracy_score(y_test, y_predict)

end_time = time.time()


"""
# from sklearn.metrics import accuracy_score

# start_time = time.time()

# # 예측
# y_predict = model.predict(x_test)

# # 확률값이므로 0.5 기준으로 반올림
# y_predict = np.round(y_predict)

# # 정확도 계산
# acc = accuracy_score(y_test, y_predict)

# end_time = time.time()

# # 출력
# print(y_predict[:10])
# print('acc_score:', acc)
# print("걸린시간:", round(end_time - start_time, 2), '초')

# accuracy_score라는 함수 이름을 변수명으로 덮어써서 나중에 재사용 불가.

# y_predict는 sigmoid 등 확률값이므로, 먼저 np.round() 해야 이진 분류에서 정확도 계산이 가능함.

# 즉, 반올림 후 accuracy_score를 계산해야 합니다.
"""