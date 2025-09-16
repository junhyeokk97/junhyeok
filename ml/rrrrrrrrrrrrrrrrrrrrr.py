import optuna
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Conv2d, Dropout2d
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn.parameter import Parameter
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation
import pennylane as qml
from datetime import datetime
import numpy as np
from tqdm import tqdm

# ----------------------- 1. 데이터 준비 -----------------------
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
train_ds = torchvision.datasets.FashionMNIST("./", train=True, download=True, transform=transform)
test_ds = torchvision.datasets.FashionMNIST("./", train=False, download=True, transform=transform)

train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
train_idx = torch.where(train_mask)[0]
train_ds.targets[train_ds.targets == 6] = 1  # 6을 1로 변경 (0/1 binary classification)

binary_train_ds = Subset(train_ds, train_idx)
train_loader = DataLoader(binary_train_ds, batch_size=1, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ----------------------- 2. QNN 모델 정의 -----------------------
class BinaryClassifier(Module):
    def __init__(self, qnn_params_init):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)
        self.fc3 = Linear(1, 1)

        self.q_device = qml.device("default.qubit", wires=2)
        self.qnn_params = Parameter(torch.tensor(qnn_params_init, dtype=torch.float32, requires_grad=True))
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

        @qml.qnode(self.q_device, interface='torch')
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(2. * x[0], wires=0)
            qml.RZ(2. * x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(2. * (torch.pi - x[0]) * (torch.pi - x[1]), wires=1)
            qml.CNOT(wires=[0, 1])
            for i in range(4):
                qml.RY(2. * self.qnn_params[2 * i], wires=0)
                qml.RY(2. * self.qnn_params[2 * i + 1], wires=1)
                qml.CNOT(wires=[0, 1])
            return qml.expval(self.obs)

        self.qnn = circuit

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        qnn_input = x[0].float()
        qnn_out = self.qnn(qnn_input).float().view(1, 1)
        x = self.fc3(qnn_out)
        logits = torch.cat([x, 1 - x], dim=1)
        return F.log_softmax(logits, dim=1)


# ----------------------- 3. Optuna 튜닝 -----------------------
def objective(trial):
    init_params = [trial.suggest_float(f"theta_{i}", 0, 2 * np.pi) for i in range(8)]
    model = BinaryClassifier(init_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = NLLLoss()

    model.train()
    losses = []
    for bidx, (data, target) in enumerate(train_loader):
        if bidx > 100:
            break
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print("✅ Best trial:", study.best_trial.params)


# ----------------------- 4. 전체 학습 (EarlyStopping 포함) -----------------------
best_params = [v for v in study.best_trial.params.values()]
bc = BinaryClassifier(best_params)
optimizer = torch.optim.Adam(bc.parameters(), lr=0.0001)
loss_func = NLLLoss()

best_val_loss = float('inf')
patience, wait = 5, 0

for epoch in range(50):  # 최대 50 epoch
    bc.train()
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for data, target in epoch_bar:
        optimizer.zero_grad()
        output = bc(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

    # Validation Loss
    bc.eval()
    val_losses = []
    with torch.no_grad():
        for data, target in val_loader:
            output = bc(data)
            val_loss = loss_func(output, target)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"📉 Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = bc.state_dict()  # ✅ best model 저장
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹ EarlyStopping at epoch {epoch}")
            break
bc.load_state_dict(best_model_state)
# ----------------------- 5. 추론 및 평가 -----------------------
bc.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Inference", leave=False):
        logits = bc(data)
        pred = logits.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

y_pred = torch.cat(all_preds).numpy().astype(int)
y_true = torch.cat(all_targets).numpy().astype(int)

test_mask = (y_true == 0) | (y_true == 6)
y_pred_mapped = np.where(y_pred == 1, 6, y_pred)
acc = (y_pred_mapped[test_mask] == y_true[test_mask]).mean()
print(f"🎯 Accuracy (labels 0/6 only): {acc:.4f}")

# ----------------------- 6. 저장 -----------------------
now = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"y_pred_{now}.csv"
np.savetxt(filename, y_pred_mapped, fmt="%d")


# 1 🎯 accuracy (labels 0/6 only): 0.8270