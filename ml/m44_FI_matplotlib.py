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

    
plot_feature_importance_datasets(model)
plt.show()

# ======= DecisionTreeClassifier =======            importances 값이 낮은 쪽은 버릴 수 있
# acc:  1.0
# [0.01481481 0.01481481 0.05530359 0.91506678]
# ======= RandomForestClassifier =======
# acc:  1.0
# [0.08915581 0.02429278 0.41193687 0.47461454]
# ======= GradientBoostingClassifier =======
# acc:  1.0
# [0.0020554  0.01358925 0.5876279  0.39672745]
#         ======= XGBClassifier =======
# acc:  1.0
# [0.0079495  0.02123976 0.77189857 0.19891217]