import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T  # (100,2)
                        # 삼성전자 종가, 하이닉스 종가.
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()
            # (100, 3)  원율, 환율, 금 시세
x3_datasets = np.array([range(100), range(300,400), range(77,177), range(33,133)]).T
            # (100, 4)
            
y = np.array(range(2001, 2101))
            # (100, )   한강의 화씨 온도.

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y,test_size=0.3, random_state=50)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu',)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(40, activation='relu')(dense3)
output1 = Dense(50, activation='relu')(dense4)
# model1 = Model(inputs = input1, outputs = output1)
# model1.summary()

 #2-2. 모델
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
output2 = Dense(30, activation='relu', name='ibm23')(dense22)
# model2 = Model(inputs = input2, outputs = output2)
# model2.summary()

#2-3. 모델
input3 = Input(shape=(4,))
dense31 = Dense(100, activation='relu', name='ibm31')(input3)
dense32 = Dense(50, activation='relu', name='ibm32')(dense31)
dense33 = Dense(50, activation='relu', name='ibm33')(dense31)
output3 = Dense(30, activation='relu', name='ibm34')(dense33)


#2-3. 모델 합치기
from keras.layers.merge import Concatenate , Concatenate
merge1 = Concatenate(name='mg1')([output1, output2, output3])
merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(10, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)
model = Model(inputs=[input1, input2, input3], outputs=last_output)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train ,validation_split=0.2, epochs=1000, batch_size=25,
          verbose=2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test,x3_test],y_test)

x1_pred = np.array([range(100,106), range(400, 406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T
y_pred = model.predict([x1_pred, x2_pred, x3_pred])
# y_pred = model.predict([x1_pred, x2_pred], y_predict)

print('loss: ', loss[0])
print('예측된 y 값 (예상치):', y_pred.flatten())


# 예측된 y 값 (예상치): [2097.8235 2100.051  2102.4192 2104.804  2107.2188 2109.6333]

# le = LabelEncoder()
# train_csv['keyword'] = le.fit_transform(train_csv['keyword'])
# train_csv['location'] = le.fit_transform(train_csv['location'])
# test_csv['keyword'] = le.fit_transform(test_csv['keyword'])
# test_csv['location'] = le.fit_transform(test_csv['location'])