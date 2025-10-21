from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets.target
y_o = y.copy()
y =np.rint(y).astype(int)       # rint 반올림
print(y)
# print(np.unique(y, return_counts=True))



x_train, x_test, y_train, y_test, y_train_o, y_test_o = train_test_split(x,y,y_o,
                                                    test_size=0.1,
                                                    random_state=50,)
                                                    # stratify=y)


### scaler는 pca 전에 하는 게 좋음.
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

############################# PCA #############################

pca = PCA(n_components=10)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
pca_EVR = pca.explained_variance_ratio_

print(np.cumsum(pca_EVR))
# [0.40217759 0.55193575 0.66896699 0.76637851 0.83268677 0.89365416
#  0.94702426 0.99130084 0.99911958]


############################# LDA #############################         y값을 알고 있기 때문에 특정 부분에 대해서는 성능이 좋지만, 과적합이 될 가능성이 있다.

# lda = LinearDiscriminantAnalysis(n_components=10)    # 50을 제외하고 세밀하게 나눈다?
# x_train = lda.fit_transform(x_train, y_train) # LDA는 y값도 같이 transform
# x_test = lda.transform(x_test)   # y_test에 transform한 값들이 들어가기 때문에 y_test는 lda를 거치지 않아도 됨?
# lda_EVR = lda.explained_variance_ratio_
# print(lda_EVR)
# print(np.cumsum(lda_EVR))
# # [0.23278024 0.35418527 0.46552727 0.56929802 0.66602084 0.75774597
# #  0.83092053 0.89539862 0.95033614 1.        ]

# print(x)
# print(x.shape)

model = RandomForestRegressor(random_state=50)

model.fit(x_train, y_train_o)

results = model.score(x_test, y_test_o)

print('score: ', results)

# lda  score:  0.39124673980416913
# pca  score:  0.4627901779777345