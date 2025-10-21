from sklearn.datasets import load_wine

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

model = XGBRegressor(random_state=seed)
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
x2 = x[['total_phenols', 'nonflavanoid_phenols', 'proanthocyanins', 'od280/od315_of_diluted_wines']]
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
print(x2_train.shape, x2_test.shape) # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

print(x_train, x_test)

model.fit(x_train, y_train)
print('FI_Drop + PCA: ', model.score(x_test, y_test))




# r2:  0.8226742921600978
# FI_Drop + PCA:  0.8226363924790552
import matplotlib.pyplot as plt

def plot_feature_importance_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), model.feature_importances_)
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

    
# plot_feature_importance_datasets(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model, importance_type='gain')          # 트리구조는 프리퀀시가 많을 수록 훈련 효율 증가?
plt.show()