import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer

# 데이터 
datasets = load_breast_cancer()
print(datasets.DESCR)   # 569개의 샘플, 30개의 피처

print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
print("############################################")
print(type(datasets))   # <class 'sklearn.utils.Bunch'>

# 데이터분리
x = datasets.data   # (569, 30) 
y = datasets.target #  (569,) 

print(x.shape, y.shape)
print(type(x))  # <class 'numpy.ndarray'> >> 0 or 1 (이진분류)
print(type(y))  # <class 'numpy.ndarray'> >> 0 or 1 (이진분류)

print(x)
# [[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
#  [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
#  [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
#  ...
#  [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
#  [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
#  [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
print(y)

# 0과 1의 개수가 몇개인지 찾아보자
# 1. pandas
# 2. numpy

# numpy
print(np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357]))

# pandas
print(pd.value_counts(y))
# 1    357
# 0    212
print(pd.Series(y).value_counts()) # >> 위랑 같은 형식
print(pd.DataFrame(y).value_counts()) # >> 위랑 같은 형식
