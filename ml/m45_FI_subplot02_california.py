from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 50
random.seed(seed)
np.random.seed(seed)

x,y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("=======", model.__class__.__name__, "=======")
    print('acc: ', model.score(x_test, y_test))
    print(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importance_datasets(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), model.feature_importances_)
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

    
plot_feature_importance_datasets(model)
plt.show()

# ======= DecisionTreeRegressor =======
# acc:  0.5633524390000897
# [0.51593086 0.0512617  0.05277013 0.02686571 0.03074463 0.13250484 0.09637703 0.09354511]

# ======= RandomForestRegressor =======
# acc:  0.7948175382140014
# [0.52009338 0.05270406 0.0445956  0.02843706 0.03125656 0.13771972 0.093358   0.09183562]

# ======= GradientBoostingRegressor =======
# acc:  0.7792459878575777
# [0.59545625 0.02976671 0.02244468 0.0045151  0.00267716 0.12410367 0.11146352 0.10957292]

# ======= XGBRegressor =======
# acc:  0.8234962888008794
# [0.4848278  0.06919315 0.04866482 0.02327505 0.02308521 0.15094434 0.09462412 0.10538556]