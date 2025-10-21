# -------------------- 1. 라이브러리 불러오기 --------------------
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib



path = './_data/dacon/smiles/'
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sample_submission.csv")


# ===== 1. SMILES 전처리 & Feature =====
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

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
        edge_index += [[start, end], [end, start]]
    data = Data(
        x=torch.tensor(atom_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous()
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data

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
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)


# ===== 2. GCN 정의 및 학습 =====
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(6, 32)
        self.conv2 = GCNConv(32, 16)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze()

train_graphs_train, train_graphs_val = train_test_split(train_graphs, test_size=0.1, random_state=190)
train_loader = DataLoader(train_graphs_train, batch_size=50, shuffle=True)
val_loader = DataLoader(train_graphs_val, batch_size=50, shuffle=False)


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

best_loss = float('inf')
patience, counter = 10, 0
for epoch in range(100):
    gnn_model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gnn_model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    gnn_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = gnn_model(batch)
            val_loss += loss_fn(out, batch.y.view(-1)).item()
    val_loss /= len(val_loader)

    print(f"[GNN] Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(gnn_model.state_dict(), './best_model_GNN.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("✅ Early Stopping!")
            break


# ===== 3. XGBoost 튜닝 =====
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=35)
best_params = study.best_params

xgb_model = XGBRegressor(**best_params)
xgb_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))


# ===== 4. 다른 모델들 =====
cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=190)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=190)
gb_model.fit(X_train, y_train)


# ===== 5. 앙상블: Stacking =====
stacking_model = StackingRegressor(
    estimators=[
        ("xgb", xgb_model),
        ("cat", cat_model),
        ("rf", rf_model),
        ("gb", gb_model)
    ],
    final_estimator=GradientBoostingRegressor()
)

stacking_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
joblib.dump(stacking_model, './best_model_Stacking_optuna8.pkl')

val_preds = stacking_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"📊 스태킹 모델 검증 RMSE: {val_rmse:.4f}")

test_pred = stacking_model.predict(X_test_final)
sub['Inhibition'] = np.round(test_pred)
sub.to_csv(path + 'submission_Stacking_optuna8.csv', index=False)
print("📄 제출 파일 저장 완료: submission_Stacking_optuna8.csv")
