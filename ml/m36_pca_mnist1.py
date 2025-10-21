from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  # (70000, 28, 28)

x = x.reshape(x.shape[0], 28*28)
print(x.shape)  # (70000, 784)

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)
# print(evr_cumsum)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(f"95% 설명하려면 필요한 주성분 수: {d}")
d1 = np.argmax(cumsum >= 0.99) + 1
print(f"95% 설명하려면 필요한 주성분 수: {d1}")
d2 = np.argmax(cumsum >= 0.999) + 1
print(f"95% 설명하려면 필요한 주성분 수: {d2}")
d3 = np.argmax(cumsum >= 1.0) + 1
print(f"95% 설명하려면 필요한 주성분 수: {d3}")

# aa = np.where(evr_cumsum>=1.0)
# aaa = np.array(aa)
# print(aaa.shape)
# print(len(aaa[0]))

# aa1 = np.where(evr_cumsum>=0.999)
# aaa1 = np.array(aa1)
# print(aaa1.shape)
# print(len(aaa1[0]))

# aa2 = np.where(evr_cumsum>=0.99)
# aaa2 = np.array(aa2)
# print(aaa2.shape)
# print(len(aaa2[0]))

# aa3 = np.where(evr_cumsum>=0.95)
# aaa3 = np.array(aa3)
# print(aaa3.shape)
# print(len(aaa3[0]))
# 1.0 몇 개 = 72
# 0.999 이상 몇 개 = 299
# 0.99 이상 몇 개 = 454
# 0.95 이상 몇 개 = 631