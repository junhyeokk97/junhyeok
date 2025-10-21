from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 50
random.seed(seed)
np.random.seed(seed)

x,y = load_iris(return_X_y=True)
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
    print('r2: ', model.score(x_test, y_test))
    print(model.feature_importances_)
   
   





   
   
   
    
# ======= DecisionTreeRegressor =======
# acc:  1.0
# [0.00740741 0.00740741 0.02765179 0.95753339]
# ======= RandomForestRegressor =======
# acc:  0.99878
# [0.00658482 0.00800233 0.43006913 0.55534372]
# ======= GradientBoostingRegressor =======
# acc:  0.9989484993677511
# [0.00655523 0.00399055 0.52438668 0.46506754]
#       ======= XGBRegressor =======
# acc:  0.9970438657269308
# [0.00928153 0.00604599 0.94663817 0.03803434]