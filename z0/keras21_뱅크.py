# https://www.kaggle.com/competitions/playground-series-s4e1/data

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split



#1 data

path = './_data/kaggle/Bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)            # [165034 x 13]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)              # [110023 x 12]
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)  # [110023 x 1]
print(train_csv, test_csv, submission_csv)
print(train_csv.head(10))
#     CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
# id
# 0     15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
# 1     15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
# 2     15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
# 3     15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
# 4     15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
# 5     15771669        Genovese          588   Germany    Male  36.0       4  131778.58              1        1.0             0.0        136024.31       1
# 6     15692819          Ch'ang          593    France  Female  30.0       8  144772.69              1        1.0             0.0         29792.11       0
# 7     15669611     Chukwuebuka          678     Spain    Male  37.0       1  138476.41              1        1.0             0.0        106851.60       0
# 8     15691707           Manna          676    France    Male  43.0       4       0.00              2        1.0             0.0        142917.13       0
# 9     15591721        Cattaneo          583   Germany    Male  40.0       4   81274.33              1        1.0             1.0        170843.07       0
print(train_csv.isna().sum())
#  CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
# id
# 0         15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
# 1         15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
# 2         15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
# 3         15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
# 4         15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
# ...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
# 165029    15667085            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
# 165030    15665521       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
# 165031    15664752            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
# 165032    15689614          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
# 165033    15732798         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
print(test_csv.isna().sum())
# CustomerId         0
# Surname            0
# CreditScore        0
# Geography          0
# Gender             0
# Age                0
# Tenure             0
# Balance            0
# NumOfProducts      0
# HasCrCard          0
# IsActiveMember     0
# EstimatedSalary    0
# Exited             0

print(train_csv.columns)
# Index(['CustomerId', '#Surname', 'CreditScore', '#', '#', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited']

from sklearn.preprocessing import LabelEncoder
for col in test_csv.columns:
    if test_csv[col].dtype == 'object':
        le = LabelEncoder()
        test_csv[col] = le.fit_transform(test_csv[col])
        # le.fit(train_csv[('Geography')])
        # train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
        # print(train_csv['Geography']. value_counts())
        
# 범주형 변수 라벨 인코딩

le_geo = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.fit_transform(test_csv['Geography'])

le_gender = LabelEncoder()
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le_gender.fit_transform(test_csv['Gender'])

# 결과 출력
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())

print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())

# 필요 없는 열 제거
cols_to_drop = ['CustomerId', 'Surname']
train_csv = train_csv.drop(columns=[col for col in cols_to_drop if col in train_csv.columns])
test_csv = test_csv.drop(columns=[col for col in cols_to_drop if col in test_csv.columns])

# 입력과 출력 분리
x = train_csv.drop(['Exited'], axis=1)
print(x.shape)          # (1675034, 10)

y = train_csv['Exited']
print(y.shape)          # (165034,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.1, random_state=11
)                              

# model
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
for _ in range(5):
    model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # > sigmoid

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])


# compile, fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping (
    monitor='val_loss', #지표
    mode='min',   # auto >> min
    patience=25,    
    restore_best_weights=True,  
)

hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,     # # over fit
        validation_split=0.2,
        callbacks=[es])

# 딕셔너리에 밸류
# loss = epochs, val_loss= epochs
print("================ hist ================")                 
print(hist)
print("================ hist.history ================")
print(hist.history)
print("================ val_loss ================")
print(hist.history['loss'])

# 그래프 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 주로 사용
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

plt.figure(figsize=(9, 6))  # 도표 사이즈 지정

plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('뱅크 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  
plt.legend(loc='upper right')  # 우측 상단 범례

plt.show()

# results, predict
results = model.evaluate(x_test, y_test)

start_time = time.time()
print(start_time)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

from sklearn.metrics import accuracy_score
y_predict = np.round(y_predict)  # 이진 분류이므로 0 또는 1로 반올림
accuracy_score = accuracy_score(y_test, y_predict)

end_time = time.time()

print("걸린시간:", round(end_time - start_time, 2), '초')

#test_csv의 예측값 
y_submit = model.predict(test_csv)  
                        # x_train.shape:(N, 8),     
y_submit = np.round(y_submit)
# print(y_submit.shape)   # (116, 1)


# submission.csv 파일 만들기

submission_csv['Exited'] = np.round(y_submit).astype(int)

# submission_csv = pd.read_csv(path + 'submission_0527_2.csv', index_col=0)
submission_csv.to_csv(path + 'submission_0527_2.csv') 


print(results)
print("loss: ", results[0])
print("acc_score:",  round(results[1], 4))
print("R2 score:", r2)
