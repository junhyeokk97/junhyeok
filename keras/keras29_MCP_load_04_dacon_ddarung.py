from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path ='./_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)

import random
r = 7275 # random.randint(1,10000)     #7275, 208, 6544, 1850, 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=r
)

from tensorflow.python.keras.models import load_model
path1 = './_save/keras28_mcp/04_dacon_ddarung/'
model = load_model(path1 + 'k28_0604_1217_0415-1894.4873.hdf5')

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)
print('Random :', r)
print('RMSE :', rmse)
print('R2 :', r2)

# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + 'submission_0526_2.csv', index=False)

# Random : 7275
# RMSE : 42.97903854097076
# R2 : 0.744493249355315
# time : 25.67926526069641 초

# Random : 7275
# RMSE : 42.97903854097076
# R2 : 0.744493249355315