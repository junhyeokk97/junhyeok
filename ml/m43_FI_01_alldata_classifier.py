from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 50
random.seed(seed)
np.random.seed(seed)

data1 = load_breast_cancer()
data2 = load_wine()
data3 = load_digits()

datasets = [data1, data2, data3]
dataset_name = ['cancer', 'wine', 'digits']

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    print("++++++++", dataset_name[i],"+++++++")

        

    for model in models:
        model.fit(x_train, y_train)
        print("=======", model.__class__.__name__, "=======")
        print('acc: ', model.score(x_test, y_test))
        # print(model.feature_importances_)
    
# ++++++++ cancer +++++++
# ======= DecisionTreeClassifier =======
# acc:  0.9473684210526315
# ======= RandomForestClassifier =======
# acc:  0.9824561403508771
# ======= GradientBoostingClassifier =======
# acc:  1.0
# ======= XGBClassifier =======
# acc:  0.9824561403508771

# ++++++++ wine +++++++
# ======= DecisionTreeClassifier =======
# acc:  0.8333333333333334
# ======= RandomForestClassifier =======
# acc:  1.0
# ======= GradientBoostingClassifier =======
# acc:  1.0
# ======= XGBClassifier =======
# acc:  1.0

# ++++++++ digits +++++++
# ======= DecisionTreeClassifier =======
# acc:  0.85
# ======= RandomForestClassifier =======
# acc:  0.9777777777777777
# ======= GradientBoostingClassifier =======
# acc:  0.9833333333333333
# ======= XGBClassifier =======
# acc:  0.9722222222222222