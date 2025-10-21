from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from tensorflow.keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50
)
from tensorflow.python.keras.models import load_model
path1 = './_save/keras28_mcp/03_diabetes/'
model = load_model(path1 + 'k28_0604_1230_0150-2865.9094.hdf5')

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)


# RMSE : 50.469586873866426
# R2 : 0.542062781751149

# RMSE : 50.35504315452177
# R2 : 0.5441391078577609