import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, MACCSkeys, DataStructs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
import joblib
import optuna
import torch
from torch_geometric.data import Data

# --- 1. 데이터 불러오기 ---
path = './_data/dacon/smiles/'
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sample_submission.csv")

# --- 2. 전처리 함수 정의 ---
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def sanitize_and_augment_smiles(smiles, n_aug=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        Chem.SanitizeMol(mol)
    except:
        return []
    augmented_smiles = []
    for _ in range(n_aug):
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
        augmented_smiles.append(rand_smiles)
    return augmented_smiles

def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'RingCount': Lipinski.RingCount(mol)
    }

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# --- 3. 데이터 정제 및 증강 ---
train = train[train['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)
test = test[test['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)
train = train.drop_duplicates(subset='Canonical_Smiles').reset_index(drop=True)

aug_smiles_list = []
aug_targets_list = []

for idx, row in train.iterrows():
    smiles = row['Canonical_Smiles']
    target = row['Inhibition']
    augmented = sanitize_and_augment_smiles(smiles, n_aug=5)
    augmented.append(smiles)
    for sm in augmented:
        aug_smiles_list.append(sm)
        aug_targets_list.append(target)

aug_train_df = pd.DataFrame({'Canonical_Smiles': aug_smiles_list, 'Inhibition': aug_targets_list})

# --- 4. GNN용 그래프 데이터 생성 ---
train_graphs = [smiles_to_graph(smi) for smi in aug_train_df['Canonical_Smiles']]
test_graphs = [smiles_to_graph(smi) for smi in test['Canonical_Smiles']]

# (이후 GNN 모델 정의 및 학습 코드 필요 시 추가 가능)

# --- 5. 피처 생성 ---
train_morgan = np.array([morgan_fp(s) for s in aug_train_df['Canonical_Smiles']])
test_morgan = np.array([morgan_fp(s) for s in test['Canonical_Smiles']])

train_maccs = np.array([maccs_fp(s) for s in aug_train_df['Canonical_Smiles']])
test_maccs = np.array([maccs_fp(s) for s in test['Canonical_Smiles']])

train_desc = aug_train_df['Canonical_Smiles'].apply(calc_descriptors).apply(pd.Series)
test_desc = test['Canonical_Smiles'].apply(calc_descriptors).apply(pd.Series)

scaler = StandardScaler()
train_desc_scaled = scaler.fit_transform(train_desc)
test_desc_scaled = scaler.transform(test_desc)

X_all = np.hstack([train_morgan, train_maccs, train_desc_scaled])
y_all = aug_train_df['Inhibition'].values

X_test_final = np.hstack([test_morgan, test_maccs, test_desc_scaled])

# --- 5. train/val 분리 ---
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=186
)

# --- 6. Optuna objective 함수 정의 ---
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 186,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 186,
        "n_jobs": -1,
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# def objective_gb(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "random_state": 186,
#     }
#     model = GradientBoostingRegressor(**params)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds))
#     return rmse

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 186,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_seed": 186,
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# --- 7. 튜닝 및 학습 함수 ---
def tune_and_train(name, objective_func):
    print(f"\n🔎 {name} 하이퍼파라미터 튜닝 시작...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=30)

    print(f"✨ {name} 최적 파라미터:")
    print(study.best_trial.params)

    best_params = study.best_trial.params
    # 재구성: Optuna에서 일부 매개변수만 나오므로, 기본값을 세팅하고 업데이트
    if name == "XGBoost":
        base_params = {
            "random_state": 186, "n_jobs": -2, "verbosity": 0
        }
        base_params.update(best_params)
        model = XGBRegressor(**base_params)
    elif name == "LightGBM":
        base_params = {
            "random_state": 186, "n_jobs": -2
        }
        base_params.update(best_params)
        model = LGBMRegressor(**base_params)
    # elif name == "GradientBoosting":
    #     base_params = {
    #         "random_state": 186
    #     }
    #     base_params.update(best_params)
    #     model = GradientBoostingRegressor(**base_params)
    elif name == "RandomForest":
        base_params = {
            "random_state": 186, "n_jobs": -2
        }
        base_params.update(best_params)
        model = RandomForestRegressor(**base_params)
    elif name == "CatBoost":
        base_params = {
            "random_seed": 186,
            "verbose": 0
        }
        base_params.update(best_params)
        model = CatBoostRegressor(**base_params)
    else:
        raise ValueError("Unknown model name")

    # train + val 전체 데이터로 재학습
    model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"✅ {name} 검증 RMSE: {val_rmse:.4f}")
    return model, val_rmse

from sklearn.ensemble import VotingRegressor

# --- 8. 모델 튜닝 및 선택 ---
models_to_tune = {
    "XGBoost": objective_xgb,
    "LightGBM": objective_lgb,
    # "GradientBoosting": objective_gb,
    "RandomForest": objective_rf,
    "CatBoost": objective_cat
}

best_score = np.inf
best_model_name = None
trained_models = {}

for name, obj_func in models_to_tune.items():
    model, score = tune_and_train(name, obj_func)
    trained_models[name] = model
    if score < best_score:
        best_score = score
        best_model_name = name

print(f"\n\U0001f389 최고 성능 모델: {best_model_name} (검증 RMSE: {best_score:.4f})")

# --- 9. 최고 단일 모델 저장 ---
joblib.dump(trained_models[best_model_name], f'./best_model_{best_model_name}_optuna.pkl')
print(f"\U0001f4be 모델 저장 완료: best_model_{best_model_name}_optuna.pkl")

# --- 10. 테스트 데이터 예측 및 제출 파일 생성 (단일 모델) ---
test_pred = trained_models[best_model_name].predict(X_test_final)
test_pred = np.round(test_pred)
sub['Inhibition'] = test_pred
sub.to_csv(path + f'submission_{best_model_name}_optuna.csv', index=False)
print(f"\U0001f4dc 제출 파일 저장 완료: submission_{best_model_name}_optuna.csv")

# --- 11. 앙상블 모델 생성 및 평가 ---
if len(trained_models) >= 2:
    print("\n\U0001f517 앙상블 모델 생성 중...")
    ensemble_model = VotingRegressor(
        estimators=[(name, model) for name, model in trained_models.items()]
    )
    ensemble_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    val_preds = ensemble_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"\n\U0001f4ca 앙상블 모델 검증 RMSE: {val_rmse:.4f}")

    # 앙상블 모델 저장
    joblib.dump(ensemble_model, './best_model_Ensemble_optuna.pkl')
    print("\U0001f4be 앙상블 모델 저장 완료: best_model_Ensemble_optuna.pkl")

    # 앙상블 테스트 예측 및 제출
    test_pred = ensemble_model.predict(X_test_final)
    test_pred = np.round(test_pred)
    sub['Inhibition'] = test_pred
    sub.to_csv(path + 'submission_Ensemble_optuna.csv', index=False)
    print("\U0001f4dc 제출 파일 저장 완료: submission_Ensemble_optuna.csv")



# 🎉 최고 성능 모델: XGBoost (검증 RMSE: 1.1812)
# 📄 제출 파일 저장 완료: submission_XGBoost_optuna.csv