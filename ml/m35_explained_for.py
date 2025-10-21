from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50,
                                                    stratify=y)

### scaler는 pca 전에 하는 게 좋음.
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

for i in range(x.shape[1]):
    pca = PCA(n_components=i+1)
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    model = RandomForestClassifier(random_state=50)

    model.fit(x_train1, y_train)

    results = model.score(x_test1, y_test)

    print(x_train1.shape, '의 score: ', results)
    # (135, 1) 의 score:  0.8666666666666667
    # (135, 2) 의 score:  0.9333333333333333
    # (135, 3) 의 score:  0.8666666666666667
    # (135, 4) 의 score:  1.0
evr = pca.explained_variance_ratio_ # 설명 가능한 변화율
print('evr: ', evr) # evr:  [0.73155643 0.22553728 0.03755841 0.00534789]   pca 자체에 미치는 영향
print('evr_sum: ', sum(evr))    # evr_sum:  1.0  >> pca 자체에 미치는 영향의 총합.

evr_cumsum = np.cumsum(evr)
print('누적합: ', evr_cumsum)   # 누적합:  [0.73155643 0.95709371 0.99465211 1.        ]

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()