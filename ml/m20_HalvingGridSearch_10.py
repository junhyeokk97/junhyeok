import numpy as np
import time
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import HalvingGridSearchCV
import joblib

# 1. 데이터 불러오기
x, y = fetch_covtype(return_X_y=True)


# 클래스 레이블이 1부터 시작하는 문제를 해결
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # 0부터 6까지의 값으로 변환


# 훈련, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=55, stratify=y
)

# 데이터 정규화
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# StratifiedKFold로 교차 검증 준비
n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=55)

# 하이퍼파라미터 그리드 설정
parameters = [
    {'n_estimators': [100, 500], 'max_depth': [6, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    {'min_child_weight': [2, 3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]}
]

# 2. 모델 정의 (XGBClassifier)
xgb = XGBClassifier()

# 3. HalvingGridSearchCV 모델 설정
model = HalvingGridSearchCV(xgb, parameters, cv=kfold, verbose=1, refit=True,
                            n_jobs=-1, random_state=33, factor=3, min_resources=20,
                            max_resources=464809)

# 4. 모델 훈련 시작
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 5. 모델 평가
print('최적의 매개변수 : ', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('best_acc_score :', accuracy_score(y_test, y_pred_best))

print("걸린 시간:", round(end - start, 2), '초')

# 모델 결과 저장
path = './_save/m15_cv_results/'
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm20_00_rs_cv_10.csv', index=False))

joblib.dump(model.best_estimator_, path + 'm20_10_best_model.joblib')
