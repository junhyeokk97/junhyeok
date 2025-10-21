from sklearn.datasets import fetch_covtype

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = fetch_covtype()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)

y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('r2: ', model.score(x_test, y_test))
print(model.feature_importances_)

print(np.percentile(model.feature_importances_, 25))

percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile)) <class 'numpy.float64'>

col_name=[]
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name) # ['AveBedrms', 'Population']

x = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x.drop(columns=col_name)
x2 = x[['Slope', 'Hillshade_3pm', 'Soil_Type_0', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_17', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_27', 'Soil_Type_33', 'Soil_Type_35']]

# print(x2)

model.fit(x_train, y_train)
print('r2_2: ', model.score(x_test, y_test))

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape) # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

print(x_train, x_test)

model.fit(x_train, y_train)
print('FI_Drop + PCA: ', model.score(x_test, y_test))




# r2_2:  0.8323637740525283
# FI_Drop + PCA:  0.834962651888059