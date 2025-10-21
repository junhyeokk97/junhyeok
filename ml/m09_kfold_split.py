import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']
print(x)
print(y)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2        
# 1                  4.9               3.0                1.4               0.2        
# 2                  4.7               3.2                1.3               0.2        
# 3                  4.6               3.1                1.5               0.2        
# 4                  5.0               3.6                1.4               0.2        
# ..                 ...               ...                ...               ...        
# 145                6.7               3.0                5.2               2.3        
# 146                6.3               2.5                5.0               1.9        
# 147                6.5               3.0                5.2               2.0        
# 148                6.2               3.4                5.4               2.3        
# 149                5.9               3.0                5.1               1.8 

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=False)

for index, (train_index , val_index) in enumerate(kfold.split(df)):
    print("=======[", index ,"]=======")
    print(train_index, '\n', val_index) 