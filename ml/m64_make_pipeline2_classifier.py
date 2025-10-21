# # cancer, dacon_당뇨병, kaggle_bank, wine, digits
# # Decision, xgb, lgbm, cat,
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_breast_cancer, load_wine, load_digits
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline

# path1 = './_data/dacon/diabetes/'
# train_csv1 = pd.read_csv(path1 + 'train.csv', index_col=0)
# test_csv1 = pd.read_csv(path1 + 'test.csv', index_col=0)

# path2 = './_data/kaggle/bank/'
# train_csv2 = pd.read_csv(path2+'train.csv', index_col=0)
# test_csv2 = pd.read_csv(path2+'test.csv', index_col=0)

# data_list = [load_breast_cancer, diabetes, bank, load_wine, load_digits]

# scl_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]

# model_list = [RandomForestClassifier(), XGBClassifier(), CatBoostClassifier(), LGBMClassifier()]

# for i, data_set in enumerate(data_list):
    
    





# x, y = load_breast_cancer(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=333, stratify=y)

# # scl = MinMaxScaler()
# # x_train = scl.fit_transform(x_train)
# # x_test = scl.transform(x_test)

# # # 2. 모델
# # model = RandomForestClassifier()

# model = make_pipeline(StandardScaler(), RandomForestClassifier())
# # model = make_pipeline(MinMaxScaler(), SVC())

# # 3. 훈련
# model.fit(x_train, y_train)

# # 4. 평가, 예측
# results = model.score(x_test, y_test)
# print('score: ', results)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print('acc: ', acc)

# -*- coding: utf-8 -*-
# 5개 데이터셋 × 4개 스케일러 × 4개 모델 정확도 비교 스크립트
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --------------------------------------------------
# 0) 경로/데이터 로드
# --------------------------------------------------
path_diabetes = Path('./_data/dacon/diabetes/')
path_bank     = Path('./_data/kaggle/bank/')

train_csv1 = pd.read_csv(path_diabetes / 'train.csv', index_col=0)
test_csv1  = pd.read_csv(path_diabetes / 'test.csv', index_col=0)   # 사용은 안 하지만, 존재 가정
train_csv2 = pd.read_csv(path_bank / 'train.csv', index_col=0)
test_csv2  = pd.read_csv(path_bank / 'test.csv', index_col=0)       # 사용은 안 하지만, 존재 가정

RANDOM_STATE = 42

# --------------------------------------------------
# 1) 타깃 탐지 & 전처리 유틸
# --------------------------------------------------
def infer_target_name(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # 후보가 없으면 마지막 컬럼을 타깃으로 가정
    return df.columns[-1]

def prepare_xy_from_csv(df: pd.DataFrame, target_candidates):
    """타깃 컬럼 탐지 → X,y 분리 → 범주형 원핫"""
    target = infer_target_name(df, target_candidates)
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    # 원핫 (범주형 자동 인코딩)
    X = pd.get_dummies(X, drop_first=False)
    return X.values, y.values, target

# --------------------------------------------------
# 2) 각 데이터셋 X,y 구성
# --------------------------------------------------
datasets = []

# (a) breast_cancer (2클래스)
X, y = load_breast_cancer(return_X_y=True)
datasets.append(("breast_cancer", X, y))

# (b) wine (3클래스)
X, y = load_wine(return_X_y=True)
datasets.append(("wine", X, y))

# (c) digits (10클래스)
X, y = load_digits(return_X_y=True)
datasets.append(("digits", X, y))

# (d) dacon_diabetes (일반적으로 이진 분류 가정; 타깃 자동 탐지)
# 흔한 후보: 'Outcome', 'outcome', 'target', 'Target', 'label', 'Label', 'diabetes'
X, y, tgt1 = prepare_xy_from_csv(
    train_csv1, target_candidates=["Outcome", "outcome", "target", "Target", "label", "Label", "diabetes"]
)
datasets.append((f"dacon_diabetes[{tgt1}]", X, y))

# (e) kaggle_bank (이진 분류 가정; 타깃 자동 탐지)
# 흔한 후보: 'y', 'deposit', 'subscribed', 'target', 'Target', 'label', 'Label', 'Exited', 'churn'
X, y, tgt2 = prepare_xy_from_csv(
    train_csv2, target_candidates=["y", "deposit", "subscribed", "target", "Target", "label", "Label", "Exited", "churn"]
)
datasets.append((f"kaggle_bank[{tgt2}]", X, y))

# --------------------------------------------------
# 3) 스케일러 & 모델 팩토리
# --------------------------------------------------
scalers = [
    ("MinMax", MinMaxScaler()),
    ("Standard", StandardScaler()),
    ("Robust", RobustScaler()),
    ("MaxAbs", MaxAbsScaler()),
]

def make_models(n_classes: int):
    """클래스 수에 따라 목적함수/파라미터 자동 설정"""
    is_multi = n_classes > 2
    xgb_params = dict(
        n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
        eval_metric="mlogloss" if is_multi else "logloss",
        objective="multi:softprob" if is_multi else "binary:logistic",
        random_state=RANDOM_STATE,
    )
    lgbm_params = dict(
        n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass" if is_multi else "binary",
        num_class=n_classes if is_multi else None,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(**xgb_params)
    lgbm = LGBMClassifier(**{k: v for k, v in lgbm_params.items() if v is not None})
    cat  = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        loss_function="MultiClass" if is_multi else "Logloss",
        verbose=0, random_state=RANDOM_STATE
    )
    return [
        ("RandomForest", rf),
        ("XGBoost", xgb),
        ("CatBoost", cat),
        ("LightGBM", lgbm),
    ]

# --------------------------------------------------
# 4) 루프 실행
# --------------------------------------------------
rows = []

for ds_name, X, y in datasets:
    n_classes = len(np.unique(y))
    models = make_models(n_classes)

    # 동일 분할 유지
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    for sc_name, scaler in scalers:
        for mdl_name, mdl in models:
            # 대부분 모델은 스케일 영향이 다르지만, 비교 일관성 위해 전부 파이프라인 적용
            pipe = make_pipeline(scaler, mdl)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            acc = accuracy_score(y_te, pred)

            rows.append({
                "dataset": ds_name,
                "n_classes": n_classes,
                "scaler": sc_name,
                "model": mdl_name,
                "accuracy": acc,
            })

# 결과 표
result = pd.DataFrame(rows).sort_values(["dataset", "accuracy"], ascending=[True, False]).reset_index(drop=True)

# 데이터셋별 Top-3 요약
topk = result.groupby("dataset").head(3).reset_index(drop=True)

pd.set_option("display.max_rows", 200)
print("\n=== 전체 결과(상위 정확도 우선, 데이터셋별 정렬) ===")
print(result)

print("\n=== 데이터셋별 TOP-3 조합 ===")
print(topk)
