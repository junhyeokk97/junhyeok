from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = load_iris()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("=======", model.__class__.__name__, "=======")
    print('acc: ', model.score(x_test, y_test))
    print(model.feature_importances_)

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

from xgboost.plotting import plot_importance    # 트리구조는 프리퀀시가 많을 수록 훈련 효율 증가? / frequency 얼마나 split을 했는지?
plot_importance(model, 
                importance_type='gain',
                #importance_type='frequency' ) >> default
                #importance_type='cover' )
                title='feature importance [gain]')          

plt.show()

"""

weight : 얼마나 자주 split 했는지, frequency
gain : split이 모델의 성능을 얼마나 개선했나 // 통상적으로 많이 씀
cover : split하기 위한 sample 수

"""