from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 50
random.seed(seed)
np.random.seed(seed)

x,y = load_breast_cancer(return_X_y=True)
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
# acc:  0.819047619047619
# [0.         0.05032817 0.         0.         0.00412537 0.007334
#  0.03442808 0.         0.         0.         0.00261062 0.
#  0.         0.         0.00109403 0.         0.         0.
#  0.         0.         0.00393131 0.04595816 0.         0.69002746
#  0.01093628 0.         0.         0.13572464 0.0055005  0.00800139]

# ======= RandomForestRegressor =======
# acc:  0.8130761904761905
# [0.00244525 0.02039537 0.00322697 0.00337872 0.00275198 0.00082254
#  0.0036824  0.15485541 0.00168004 0.00054879 0.00347656 0.00143557
#  0.00529971 0.01025255 0.00322975 0.0013005  0.00465839 0.00267213
#  0.00141318 0.00432723 0.10021797 0.03238268 0.2069637  0.13302339
#  0.0104389  0.00362774 0.01228417 0.26337333 0.00259432 0.00324076]

# ======= GradientBoostingRegressor =======
# acc:  0.7934572512012639
# [5.71786782e-05 3.50927933e-02 1.47601055e-03 8.70602062e-04
#  6.84557845e-04 1.79930419e-03 5.87468914e-03 2.15831739e-01
#  5.43742105e-04 1.98070747e-03 1.63687211e-03 6.04778050e-04
#  1.13275653e-03 1.30850040e-02 8.62831195e-04 2.00756208e-03
#  3.21116321e-03 4.04903148e-03 3.23555548e-04 3.66790440e-03
#  3.47767310e-01 3.53129303e-02 7.53399357e-03 1.63984465e-01
#  3.93399955e-03 1.78071056e-03 2.10037281e-02 1.20717408e-01
#  9.39523083e-04 2.23314915e-03]

# ======= XGBRegressor =======
# acc:  0.8471683263778687
# [2.6998422e-03 1.5960600e-02 6.7851967e-03 2.0776405e-04 5.1515945e-03
#  5.4325792e-03 2.3788521e-02 1.5823158e-03 2.1367903e-04 6.1321232e-05
#  5.2994117e-03 9.4768911e-06 1.0227027e-03 4.1427962e-03 1.4145334e-03
#  7.2151277e-05 1.3238216e-04 1.2309265e-03 1.4107698e-03 2.6467454e-04
#  8.2128616e-03 2.3332920e-02 2.2717020e-02 7.6493675e-01 2.3847674e-03
#  2.0627924e-05 9.3011744e-03 9.1575064e-02 6.2612630e-04 9.4122679e-06]