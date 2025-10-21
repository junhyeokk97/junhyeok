from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# LDA의 n_component는 y label 갯수 -1 이하로 만들 수 있다.
lda = LinearDiscriminantAnalysis(n_components=1)    # 1을 제외하고 세밀하게 나눈다?
train_lda = lda.fit_transform(x_train, y_train) # LDA는 y값도 같이 transform
test_lda = lda.transform(x_test)   # y_test에 transform한 값들이 들어가기 때문에 y_test는 lda를 거치지 않아도 됨?
# print(x)
# print(x.shape)


model = RandomForestClassifier(random_state=50)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)

print(x.shape, '의 score: ', results)