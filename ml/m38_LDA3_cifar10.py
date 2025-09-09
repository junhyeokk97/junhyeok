from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_train = y_train.ravel()
y_test = y_test.ravel()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

### scaler는 pca 전에 하는 게 좋음.
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

# LDA의 n_component는 y label 갯수 -1 이하로 만들 수 있다.
lda = LinearDiscriminantAnalysis(n_components=5)    # 50을 제외하고 세밀하게 나눈다?
train_lda = lda.fit_transform(x_train, y_train) # LDA는 y값도 같이 transform
test_lda = lda.transform(x_test)   # y_test에 transform한 값들이 들어가기 때문에 y_test는 lda를 거치지 않아도 됨?
# print(x)
# print(x.shape)


model = RandomForestClassifier(random_state=50)

model.fit(train_lda, y_train)

results = model.score(test_lda, y_test)

print('score: ', results)

# score:  0.3253