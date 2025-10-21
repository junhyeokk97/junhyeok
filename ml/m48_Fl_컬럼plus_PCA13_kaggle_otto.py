from sklearn.datasets import load_diabetes

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/kaggle/otto/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sampleSubmission.csv')

# 3. 데이터 분리
x = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

# 4. 라벨 인코딩 및 스케일링
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0 ~ 8 정수 인코딩
y = y_encoded.reshape(-1, 1)


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
print(col_name) # ['feat_4', 'feat_5', 'feat_6', 'feat_10', 'feat_12', 'feat_18', 'feat_20', 'feat_21', 'feat_22', 'feat_28', 'feat_31', 'feat_33', 'feat_44', 'feat_46', 'feat_49', 'feat_61', 'feat_63', 'feat_65', 'feat_66', 'feat_70', 'feat_73', 'feat_74', 'feat_82', 'feat_89']

x = pd.DataFrame(x, columns=x.columns)
x1 = x.drop(columns=col_name)
x2 = x[['feat_4', 'feat_5', 'feat_6', 'feat_10', 'feat_12', 'feat_18', 'feat_20', 'feat_21', 'feat_22', 'feat_28', 'feat_31', 'feat_33', 'feat_44', 'feat_46', 'feat_49', 'feat_61', 'feat_63', 'feat_65', 'feat_66', 'feat_70', 'feat_73', 'feat_74', 'feat_82', 'feat_89']]
# print(x2)

model.fit(x_train, y_train)
print('r2_2: ', model.score(x_test, y_test))

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

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




# r2_2:  0.8172268907563025
# FI_Drop + PCA:  0.8154492566257272