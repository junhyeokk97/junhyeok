from sklearn.datasets import load_diabetes

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/kaggle/santander/'

  # 실제 sample_submission.csv 위치
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 2. 피처, 타겟 분리
x = train.drop(columns=['target', 'ID_code'])
y = train['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('r2: ', model.score(x_test, y_test))
print(model.feature_importances_)   

# print(np.percentile(model.feature_importances_, 25))

percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile)) <class 'numpy.float64'>

col_name=[]
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
print(col_name) # ['var_3', 'var_4', 'var_7', 'var_8', 'var_10', 'var_14', 'var_15', 'var_16', 'var_17', 'var_19', 'var_20', 'var_27', 'var_30', 'var_41', 'var_42', 'var_47', 'var_50', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_65', 'var_66', 'var_68', 'var_69', 'var_73', 'var_79', 'var_83', 'var_84', 'var_96', 'var_97', 'var_102', 'var_113', 'var_117', 'var_124', 'var_136', 'var_138', 'var_140', 'var_142', 'var_144', 'var_152', 'var_153', 'var_159', 'var_160', 'var_161', 'var_168', 'var_182', 'var_189']

x = pd.DataFrame(x, columns=x.columns)
x1 = x.drop(columns=col_name)
x2 = x[['var_3', 'var_4', 'var_7', 'var_8', 'var_10', 'var_14', 'var_15', 'var_16', 'var_17', 'var_19', 'var_20', 'var_27', 'var_30', 'var_41', 'var_42', 'var_47', 'var_50', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_65', 'var_66', 'var_68', 'var_69', 'var_73', 'var_79', 'var_83', 'var_84', 'var_96', 'var_97', 'var_102', 'var_113', 'var_117', 'var_124', 'var_136', 'var_138', 'var_140', 'var_142', 'var_144', 'var_152', 'var_153', 'var_159', 'var_160', 'var_161', 'var_168', 'var_182', 'var_189']]
# print(x2)

model.fit(x_train, y_train)
print('r2_2: ', model.score(x_test, y_test))

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

# print(x1_train.shape, x1_test.shape)
# print(x2_train.shape, x2_test.shape)
# print(y_train.shape, y_test.shape)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
# print(x2_train.shape, x2_test.shape)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

# print(x_train, x_test)

model.fit(x_train, y_train)
print('FI_Drop + PCA: ', model.score(x_test, y_test))




# r2_2:  0.9121
# FI_Drop + PCA:  0.8988