# -------------------- 1. 라이브러리 불러오기 --------------------
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs, Crippen, QED, Lipinski, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import random
seed = 50
random.seed(seed)
np.random.seed(seed)
# -------------------- 2. 데이터 불러오기 --------------------
print("📦 Loading data...")
path = './_data/dacon/smiles/'
train = pd.read_csv(path +'train.csv')
test = pd.read_csv(path +'test.csv')
sub = pd.read_csv(path +'submission.csv')

y = train['Inhibition'].values

# -------------------- 3. 전처리 함수 정의 --------------------
desc_names = [d[0] for d in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def smiles_to_mol(smi):
    try: return Chem.MolFromSmiles(smi)
    except: return None

def mol_to_morgan_fp(mol, radius=2, nBits=2048):
    arr = np.zeros((nBits,), dtype=np.int8)
    if mol: DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits), arr)
    return arr

def mol_to_maccs(mol):
    arr = np.zeros((167,), dtype=np.int8)
    if mol: DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
    return arr

def mol_to_descriptors(mol):
    if mol is None: return [0] * len(desc_names)
    try: return list(desc_calc.CalcDescriptors(mol))
    except: return [0] * len(desc_names)

def mol_to_physchem(mol):
    if mol is None:
        return [0]*6
    return [
        Crippen.MolLogP(mol),
        QED.qed(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcTPSA(mol)
    ]

def featurize_smiles(smiles_list):
    features = []
    for smi in tqdm(smiles_list, desc="🔬 Featurizing"):
        mol = smiles_to_mol(smi)
        feat = np.concatenate([
            mol_to_morgan_fp(mol),
            mol_to_maccs(mol),
            mol_to_descriptors(mol),
            mol_to_physchem(mol)
        ])
        features.append(feat)
    return np.array(features)

# -------------------- 4. Featurization --------------------
print("⚙️  Featurizing train data...")
X_train = featurize_smiles(train['Canonical_Smiles'])
print("⚙️  Featurizing test data...")
X_test = featurize_smiles(test['Canonical_Smiles'])

# -------------------- 5. 전처리: 이상치 제거, 결측값 처리, 스케일링 --------------------
print("🧹 Preprocessing: 이상치 제거(Z-score < 5)...")
z_scores = np.abs(zscore(X_train, nan_policy='omit'))
mask = (z_scores < 5).all(axis=1)

print(f"📊 이상치 제거 전 데이터 개수: {len(X_train)}")
print(f"📉 이상치 제거 후 데이터 개수: {mask.sum()}")

# 최소 샘플 체크
if mask.sum() == 0:
    print("⚠️ 이상치 제거 후 샘플이 없습니다. 스킵합니다.")
else:
    X_train = X_train[mask]
    y = y[mask]

print("🧼 결측값 처리 및 스케일링 (RobustScaler)...")
X_train = pd.DataFrame(X_train).fillna(X_train.mean()).values
X_test = pd.DataFrame(X_test).fillna(X_train.mean()).values

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------- 6. Feature Selection --------------------
print("🧠 Feature Selection using XGBoost...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.1, random_state=seed)

xgb_fs = XGBRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
xgb_fs.fit(X_tr, y_tr)

booster = xgb_fs.get_booster()
gain_importance = booster.get_score(importance_type='gain')
total_gain = sum(gain_importance.values())
feature_scores = [gain_importance.get(f"f{i}", 0) / total_gain for i in range(X_tr.shape[1])]
thresholds = np.sort(feature_scores)

best_rmse = float('inf')
best_mask = None

for thresh in tqdm(thresholds, desc="🔍 Searching best feature subset"):
    selector = SelectFromModel(xgb_fs, threshold=thresh, prefit=True)
    X_tr_sel = selector.transform(X_tr)
    X_val_sel = selector.transform(X_val)
    if X_tr_sel.shape[1] == 0:
        continue
    model_tmp = XGBRegressor(n_estimators=300, early_stopping_rounds=30, eval_set=[(X_val_sel, y_val)], verbose=False)
    model_tmp.fit(X_tr_sel, y_tr)
    preds = model_tmp.predict(X_val_sel)
    rmse = mean_squared_error(y_val, preds, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_mask = selector.get_support()

print(f"✅ 최적 feature 개수: {sum(best_mask)} | 최적 RMSE: {best_rmse:.4f}")

X_train_sel = X_train[:, best_mask]
X_test_sel = X_test[:, best_mask]

# -------------------- 7. model fit --------------------
print("📈 Training models with early stopping...")

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, subsample=0.8,
                   early_stopping_rounds=30, eval_set=[(X_val[:, best_mask], y_val)], verbose=False)

lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4,
                    early_stopping_round=30, eval_set=[(X_val[:, best_mask], y_val)], verbose=False)

cat = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=4,
                        early_stopping_rounds=30, eval_set=(X_val[:, best_mask], y_val), verbose=0)

stack = StackingRegressor(
    estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
    final_estimator=Ridge(),
    passthrough=True,
    n_jobs=-1
)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def pearson_correlation(y_true, y_pred):
    return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

def competition_score(y_true, y_pred):
    return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

stack.fit(X_train_sel, y)
now = datetime.now().strftime("%Y-%m-%d_%H%M")

# 모델 저장
model_path = f'./stacking_model_{now}.pkl'
joblib.dump(stack, model_path)
print(f"✅ 모델 학습 및 저장 완료: {model_path}")

# -------------------- 8. 제출 파일 생성 --------------------
print("📁 Generating submission...")
test_preds = stack.predict(X_test_sel)
sub['Inhibition'] = test_preds

submission_path = f'./submission_{now}.csv'
sub.to_csv(submission_path, index=False)
print(f"🎯 제출 파일 저장 완료: {submission_path}")
