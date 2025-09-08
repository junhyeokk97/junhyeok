from sklearn.datasets import fetch_california_housing

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = fetch_california_housing()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

model = XGBRegressor(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('r2: ', model.score(x_test, y_test))
print(model.feature_importances_)

model.fit(x_train, y_train)
print('FI_Drop + PCA: ', model.score(x_test, y_test))

