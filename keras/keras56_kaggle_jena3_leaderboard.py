import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

csv = "jena_최준혁4_submit.csv"

path1 = "C:/study25/_data/kaggle/jena/"
path2 = "C:/study25/_save/keras56/"

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)

y_answer = datasets.iloc[-144:,-1]
print(y_answer)
print(y_answer.shape)

z = pd.read_csv(path2 + csv, index_col=0)
print(z)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
rmse = RMSE(y_answer, z)
print('d: ', rmse)