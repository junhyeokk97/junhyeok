import pandas as pd
import numpy as np
import torch
from rdkit.Chem import rdMolDescriptors
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, MACCSkeys, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
import joblib
import optuna
from sklearn.decomposition import PCA
import random
seed = 50
random.seed(seed)
np.random.seed(seed)
# 1. 데이터 로드
path = './_data/dacon/smiles/'
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sample_submission.csv")

# 2. SMILES 전처리 및 피처 생성 함수
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import numpy as np

def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic())
    ]

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

def smiles_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    data = Data(
        x=torch.tensor(atom_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous()
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data

# 3. 전처리 및 데이터 준비
train = train[train['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)
test = test[test['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)

train_graphs = [smiles_to_graph(s, y) for s, y in zip(train['Canonical_Smiles'], train['Inhibition'])]
test_graphs = [smiles_to_graph(s) for s in test['Canonical_Smiles']]

train_morgan = np.array([morgan_fp(s) for s in train['Canonical_Smiles']])
test_morgan = np.array([morgan_fp(s) for s in test['Canonical_Smiles']])

train_maccs = np.array([maccs_fp(s) for s in train['Canonical_Smiles']])
test_maccs = np.array([maccs_fp(s) for s in test['Canonical_Smiles']])

train_desc = train['Canonical_Smiles'].apply(calc_descriptors).apply(pd.Series)
test_desc = test['Canonical_Smiles'].apply(calc_descriptors).apply(pd.Series)

scaler = StandardScaler()
train_desc_scaled = scaler.fit_transform(train_desc)
test_desc_scaled = scaler.transform(test_desc)

X_all = np.hstack([train_morgan, train_maccs, train_desc_scaled])
y_all = train['Inhibition'].values
X_test_final = np.hstack([test_morgan, test_maccs, test_desc_scaled])

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=50)

pca = PCA(n_components=300)
train_morgan_pca = pca.fit_transform(train_morgan)
test_morgan_pca = pca.transform(test_morgan)

# 피처 합치기
X_all = np.hstack([train_morgan_pca, train_maccs, train_desc_scaled])
X_test_final = np.hstack([test_morgan_pca, test_maccs, test_desc_scaled])

# 4. GNN 모델 정의
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GATNet(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels // 2, heads=1, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        return self.lin(x).squeeze()

# 5. GNN 학습
gnn_model = GATNet()
if __name__ == "__main__":
    from torch.cuda.amp import autocast, GradScaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = gnn_model.to(device)
    loader = DataLoader(train_graphs, batch_size=64, shuffle=True, num_workers=4,)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    gnn_model.train()
    for epoch in range(74):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)  # non_blocking으로 데이터 이동 속도 향상
            optimizer.zero_grad()
            out = gnn_model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[GNN] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")



# 6. 다른 모델 학습
cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=50)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=50)
gb_model.fit(X_train, y_train)


# 7. XGBoost 튜닝 및 학습

def objective(trial):
    # XGBoost 하이퍼파라미터
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0, 1),
        "random_state": seed,
        "n_jobs": -1
    }
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))

    # CatBoost 하이퍼파라미터
    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 100, 300),
        "depth": trial.suggest_int("cat_depth", 4, 10),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.001, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1, 10),
        "random_seed": seed,
        "verbose": 0
    }
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_val)
    cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))

    # RandomForest 하이퍼파라미터
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 4),
        "random_state": seed,
        "n_jobs": -1
    }
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))

    # GradientBoosting 하이퍼파라미터
    gb_params = {
        "n_estimators": trial.suggest_int("gb_n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("gb_learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("gb_max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("gb_min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("gb_min_samples_leaf", 1, 4),
        "random_state": seed
    }
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))

    # 네 모델 RMSE 평균을 objective로 사용 (혹은 min값 등 자유롭게 수정 가능)
    avg_rmse = (xgb_rmse + cat_rmse + rf_rmse + gb_rmse) / 4
    return avg_rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(trial.params)

# 각 모델을 best trial 파라미터로 다시 학습
xgb_best = XGBRegressor(
    n_estimators=trial.params["xgb_n_estimators"],
    max_depth=trial.params["xgb_max_depth"],
    learning_rate=trial.params["xgb_learning_rate"],
    subsample=trial.params["xgb_subsample"],
    colsample_bytree=trial.params["xgb_colsample_bytree"],
    reg_alpha=trial.params["xgb_reg_alpha"],
    reg_lambda=trial.params["xgb_reg_lambda"],
    random_state=42,
    n_jobs=-1,
)
cat_best = CatBoostRegressor(
    iterations=trial.params["cat_iterations"],
    depth=trial.params["cat_depth"],
    learning_rate=trial.params["cat_learning_rate"],
    l2_leaf_reg=trial.params["cat_l2_leaf_reg"],
    random_seed=42,
    verbose=0,
)
rf_best = RandomForestRegressor(
    n_estimators=trial.params["rf_n_estimators"],
    max_depth=trial.params["rf_max_depth"],
    min_samples_split=trial.params["rf_min_samples_split"],
    min_samples_leaf=trial.params["rf_min_samples_leaf"],
    random_state=42,
    n_jobs=-1,
)
gb_best = GradientBoostingRegressor(
    n_estimators=trial.params["gb_n_estimators"],
    learning_rate=trial.params["gb_learning_rate"],
    max_depth=trial.params["gb_max_depth"],
    min_samples_split=trial.params["gb_min_samples_split"],
    min_samples_leaf=trial.params["gb_min_samples_leaf"],
    random_state=42,
)

# 전체 train 데이터로 다시 학습
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

xgb_best.fit(X_train_val, y_train_val)
cat_best.fit(X_train_val, y_train_val)
rf_best.fit(X_train_val, y_train_val)
gb_best.fit(X_train_val, y_train_val)

# 앙상블 조합
from sklearn.ensemble import VotingRegressor

ensemble_model = VotingRegressor(
    estimators=[
        ("xgb", xgb_best),
        ("cat", cat_best),
        ("rf", rf_best),
        ("gb", gb_best),
    ]
)
ensemble_model.fit(X_train_val, y_train_val)


from datetime import datetime

now = datetime.now().strftime("%m%d_%H%M")

# GNN 모델 저장
gnn_filename = f'./best_model_GNN_{now}.pt'
torch.save(gnn_model.state_dict(), gnn_filename)
print(f"✅ GNN 모델 저장 완료: {gnn_filename}")

# 앙상블 모델 학습 (전체 train+val 데이터 사용)
ensemble_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

# 앙상블 모델 저장
ensemble_filename = f'./best_model_Ensemble_optuna_{now}.pkl'
joblib.dump(ensemble_model, ensemble_filename)
print(f"✅ 앙상블 모델 저장 완료: {ensemble_filename}")

# 제출 파일 저장
submission_filename = path + f'submission_Ensemble_optuna_{now}.csv'
test_pred = ensemble_model.predict(X_test_final)
sub['Inhibition'] = np.round(test_pred)
sub.to_csv(submission_filename, index=False)
print(f"📄 제출 파일 저장 완료: {submission_filename}")

# 10. 검증 RMSE 출력
val_preds = ensemble_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"📊 앙상블 모델 검증 RMSE: {val_rmse:.4f}")





#  앙상블 모델 검증 RMSE: 12.7148         04

# 📊 앙상블 모델 검증 RMSE: 12.4340       06

#  앙상블 모델 검증 RMSE: 13.6076         07

# 📊 앙상블 모델 검증 RMSE: 12.3585       08











import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.feature_selection import SelectFromModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

# 설정
seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

# 경로
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') \
    else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')

# 데이터 로드
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# --- Feature Extraction ---
def get_molecule_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 2233
        basic = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol), Descriptors.NumAliphaticRings(mol),
            Lipinski.NumAromaticHeterocycles(mol), Lipinski.NumSaturatedHeterocycles(mol),
            Lipinski.NumAliphaticHeterocycles(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol), Descriptors.NOCount(mol), Descriptors.NHOHCount(mol),
            Descriptors.NumRadicalElectrons(mol)
        ]
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
        return basic + morgan + maccs
    except:
        return [0] * 2233

# 피처 생성
train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

x_raw = np.array(train['features'].tolist())
y = train['Inhibition'].values
x_test_raw = np.array(test['features'].tolist())

# 데이터 분할
x_train_raw, x_val_raw, y_train, y_val = train_test_split(x_raw, y, test_size=0.2, random_state=seed)

# 정규화
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train_raw)
x_val = scaler.transform(x_val_raw)
x_test = scaler.transform(x_test_raw)

# 평가 지표
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def pearson_correlation(y_true, y_pred):
    return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

def competition_score(y_true, y_pred):
    return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

# --- XGBoost 중요도 기반 Feature Selection ---
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=seed,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0
)

xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
              callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])

booster = xgb_model.get_booster()
score_dict = booster.get_score(importance_type='gain')
total_gain = sum(score_dict.values())
score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]
thresholds = np.sort(score_list)

feature_names = [f"f{i}" for i in range(x_train.shape[1])]
delete_columns = []
max_score = -np.inf

for threshold in thresholds:
    selection = SelectFromModel(xgb_model, threshold=threshold, prefit=True)
    selected_train = selection.transform(x_train)
    selected_val = selection.transform(x_val)
    if selected_train.shape[1] == 0:
        continue
    temp_model = xgb.XGBRegressor(**xgb_model.get_params())
    temp_model.fit(selected_train, y_train, eval_set=[(selected_val, y_val)], verbose=0)
    score = competition_score(y_val, temp_model.predict(selected_val))
    if score > max_score:
        max_score = score
        best_selection = selection
        delete_columns = [feature_names[i] for i, selected in enumerate(selection.get_support()) if not selected]

print(f"\n🔥 Best score after feature selection: {max_score:.4f}")
print(f"❌ Deleted features: {delete_columns}")

# 최종 데이터 준비
x_train_selected = best_selection.transform(scaler.fit_transform(x_raw))
x_test_selected = best_selection.transform(scaler.transform(x_test_raw))
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_train_selected, y, test_size=0.2, random_state=seed)

# --- 모델 정의 및 학습 ---
base_models = {
    "XGBoost": xgb.XGBRegressor(**xgb_model.get_params()),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=seed),
    "CatBoost": cat.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=seed, verbose=0)
}

trained_models = {}
best_score = -np.inf
best_model_name = None

for name, model in base_models.items():
    print(f"\n{name} 모델 학습 중...")
    m = copy.deepcopy(model)
    if name == "CatBoost":
        m.fit(x_train_final, y_train_final, eval_set=(x_val_final, y_val_final), early_stopping_rounds=50, verbose=0)
    elif name == "XGBoost":
        m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)],
              callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
    elif name == "LightGBM":
        m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)],
              callbacks=[lgb.early_stopping(50)])
    else:
        m.fit(x_train_final, y_train_final)

    y_pred = m.predict(x_val_final)
    score = competition_score(y_val_final, y_pred)
    print(f"→ Score: {score:.4f}")
    trained_models[name] = m
    if score > best_score:
        best_score = score
        best_model_name = name

# 스태킹
stacking_model = StackingRegressor(
    estimators=[(k.lower(), v) for k, v in trained_models.items()],
    final_estimator=Ridge(), n_jobs=-1
)

stacking_model.fit(x_train_final, y_train_final)
y_pred_stack = stacking_model.predict(x_val_final)
stack_score = competition_score(y_val_final, y_pred_stack)
print(f"→ Stacking | Score: {stack_score:.4f}")

# 최종 선택
final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]

final_model.fit(x_train_selected, y)
y_pred_test = final_model.predict(x_test_selected)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_final_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# --- [원본 코드 복원] 모든 모델 성능 비교 ---
print("\n📊 모델별 성능 비교")
for name, model in trained_models.items():
    y_pred_compare = model.predict(x_val)
    print(f"{name:20} | RMSE: {rmse(y_val, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val, y_pred_compare):.4f} | Score: {competition_score(y_val, y_pred_compare):.4f}")

print(f"{'StackingRegressor':20} | RMSE: {rmse(y_val, y_pred_stack):.4f} | NRMSE: {normalized_rmse(y_val, y_pred_stack):.4f} | Pearson: {pearson_correlation(y_val, y_pred_stack):.4f} | Score: {stack_score:.4f}")



	# 1.	RDKit 기반 피처 생성 (basic + Morgan + MACCS)
	# 2.	정규화 (StandardScaler) 적용 후 feature importance 계산
	# 3.	XGBoost gain 기준 feature importance → SelectFromModel으로 최적 threshold 탐색
	# 4.	최적 threshold 기준으로 불필요한 feature 제거
	# 5.	그 결과를 반영하여 XGBoost, LightGBM, CatBoost 등 개별 모델 학습
	# 6.	스태킹 모델 학습 및 최종 성능 비교
	# 7.	테스트셋 예측값 제출 저장까지 포함