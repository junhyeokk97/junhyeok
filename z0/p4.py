from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape)  # (442, 10)
print(y.shape)  # (442,)

# r2 > 0.62

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=10)

model = Sequential()
model.add(Dense(11, input_dim=10))
model.add(Dense(15))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])


r2= r2_score(y_test, results)
print('loss: ', loss)
print('r2 size: ', r2)



