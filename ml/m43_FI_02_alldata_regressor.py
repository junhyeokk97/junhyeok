from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 50
random.seed(seed)
np.random.seed(seed)

data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataset_name = ['california', 'diabetes']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    print("++++++++", dataset_name[i],"+++++++")

        

    for model in models:
        model.fit(x_train, y_train)
        print("=======", model.__class__.__name__, "=======")
        print('r2: ', model.score(x_test, y_test))
        # print(model.feature_importances_)
    
# ++++++++ california +++++++
# ======= DecisionTreeRegressor =======
# r2:  0.5647735304737043
# ======= RandomForestRegressor =======
# r2:  0.794655924602091
# ======= GradientBoostingRegressor =======
# r2:  0.7792230072553389
# ======= XGBRegressor =======
# r2:  0.8234962888008794


# ++++++++ diabetes +++++++
# ======= DecisionTreeRegressor =======
# r2:  -0.5238833739707429
# ======= RandomForestRegressor =======
# r2:  0.39124673980416913
# ======= GradientBoostingRegressor =======
# r2:  0.5125630008580777
# ======= XGBRegressor =======
# r2:  0.2980064759765042