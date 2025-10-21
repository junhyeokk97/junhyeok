import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target
print(x)
print(y)
print(x.shape)  #
print(y.shape)  #
