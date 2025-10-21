from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 50
random.seed(seed)
np.random.seed(seed)

x,y = load_wine(return_X_y=True)
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
# acc:  0.7954545454545454
# [0.00238855 0.00962102 0.         0.         0.00781708 0.
#  0.62691059 0.         0.         0.03881585 0.0876582  0.
#  0.22678871]

# ======= RandomForestRegressor =======
# acc:  0.9102352272727273
# [3.99287838e-02 4.93776635e-03 1.99923150e-03 3.14877783e-03
#  4.64959456e-03 1.44517305e-03 3.70334621e-01 2.71045484e-04
#  8.82866778e-04 7.43195071e-02 2.83806757e-02 2.19350518e-01
#  2.50351439e-01]

# ======= GradientBoostingRegressor =======
# acc:  0.8400214892779644
# [1.09862998e-02 2.84303323e-03 2.39565607e-05 5.33914888e-04
#  5.82396902e-04 3.10490326e-04 6.31251307e-01 1.10034613e-05
#  1.42512406e-05 5.51509209e-02 8.31810822e-02 5.50157405e-05
#  2.15056327e-01]

# ======= XGBRegressor =======
# acc:  0.8104515671730042
# [5.4639345e-03 1.2369441e-03 2.1467084e-02 4.6647237e-05 3.9382708e-06
#  4.2263468e-07 5.6894308e-01 5.0789049e-07 1.3640754e-06 1.2320421e-01
#  1.2145415e-03 6.7207935e-07 2.7841666e-01]