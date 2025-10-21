import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 데이터 로딩
x, y = fetch_covtype(return_X_y=True)

# Train/Test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=50, stratify=y
)

# LabelEncoder 사용: y를 0부터 시작하게 변환
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 모델 인스턴스 생성
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
rf = RandomForestClassifier(verbose=0)
# cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbose=0)

models = [xgb, rf, lg]

train_list = []
test_list = []

# 모델 학습 및 예측 저장
for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append(y_test_pred)
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} acc: {1:.4f}'.format(class_name, score))

# 스태킹용 데이터 구성
x_train_new = np.array(train_list).T
x_test_new = np.array(test_list).T

# 메타 모델 (스태킹 최종 모델)
model2 = RandomForestClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)

# 성능 출력
score2 = accuracy_score(y_test, y_pred2)
print('stacking acc:', score2)



