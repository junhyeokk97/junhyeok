from sklearn.datasets import fetch_california_housing

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/kaggle/bank/'
train_df = pd.read_csv(path+'train.csv', index_col=0)
test_df = pd.read_csv(path+'test.csv', index_col=0)
submission_df = pd.read_csv(path+'sample_submission.csv')

le = LabelEncoder()
train_df['Geography'] = le.fit_transform(train_df['Geography'])
test_df['Geography'] = le.transform(test_df['Geography'])
train_df['Gender'] = le.fit_transform(train_df['Gender'])
test_df['Gender'] = le.transform(test_df['Gender'])

train_df = train_df.drop(columns=['CustomerId','Surname'], axis=1)
test_df = test_df.drop(columns=['CustomerId','Surname'], axis=1)

x = train_df.drop(columns=['Exited'], axis=1)
y = train_df['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

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
        col_name.append(x.columns[i])
    else:
        continue
print(col_name) # ['Insulin', 'DiabetesPedigreeFunction']

x = pd.DataFrame(x, columns=x.columns)
x1 = x.drop(columns=col_name)
x2 = x[['CreditScore', 'Tenure', 'EstimatedSalary']]
# print(x2)

model.fit(x_train, y_train)
print('r2_2: ', model.score(x_test, y_test))

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
# print(x2_train.shape, x2_test.shape)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

# print(x_train, x_test)

model.fit(x_train, y_train)
print('FI_Drop + PCA: ', model.score(x_test, y_test))




# r2_2:  0.8662142510906446
# FI_Drop + PCA:  0.8620334464372273