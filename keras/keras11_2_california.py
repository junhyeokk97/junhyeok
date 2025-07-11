import ssl
# import certifi

ssl._create_default_https_context = ssl._create_unverified_context


import sklearn as sk
print(sk.__version__)       # 1.1.3
import tensorflow as tf
print(tf.__version__)       # 2.9.3
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
dataset = fetch_california_housing()
print(dataset)          # y 데이터는 타겟데이터
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (20640, 8)
print(y)
print(y.shape)      # (20640,)

# [실습]  r2 > 0.59

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=111)

model = Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(100))
model.add(Dense(1500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=100)


loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)

print('r2 score: ', r2)






# model.add(Dense(20, input_dim=8))
# model.add(Dense(456))
# model.add(Dense(112))
# model.add(Dense(59))
# model.add(Dense(10))
# model.add(Dense(1))         epochs=200, batch_size=1              r2 score:  0.4888173215255407      



# model.add(Dense(20, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(1000))
# model.add(Dense(200))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))          epochs=300, batch_size=200          r2 score:  0.2252376353276584   



# model.add(Dense(20, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(1000))
# model.add(Dense(200))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))          epochs=300, batch_size=300          r2 score:  0.3140381239073615




# model.add(Dense(20, input_dim=8))                     random 111
# model.add(Dense(100))
# model.add(Dense(1000))
# model.add(Dense(200))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))          epochs=500, batch_size=300          r2 score:  0.4657397044957253


# model.add(Dense(20, input_dim=8))                     random 111
# model.add(Dense(100))
# model.add(Dense(1000))
# model.add(Dense(200))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))          epochs=500, batch_size=500          r2 score:  0.4993622873643594
                                                              #    r2 score:  0.5034127874613198
                                                              #    r2 score:  0.514564716353439