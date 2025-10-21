import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pennylane as qml
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. 데이터 로딩 및 필터링
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

target_map = {0: 0, 6: 1}
train_mask = (train_dataset.targets == 0) | (train_dataset.targets == 6)
test_mask = (test_dataset.targets == 0) | (test_dataset.targets == 6)

train_indices = torch.where(train_mask)[0]
test_indices = torch.where(test_mask)[0]

train_targets_np = train_dataset.targets[train_indices].numpy()
train_targets_np = np.vectorize(target_map.get)(train_targets_np)
train_dataset.targets[train_indices] = torch.tensor(train_targets_np, dtype=torch.long)

test_targets_np = test_dataset.targets[test_indices].numpy()
test_targets_np = np.vectorize(target_map.get)(test_targets_np)
test_dataset.targets[test_indices] = torch.tensor(test_targets_np, dtype=torch.long)

train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 2. QNN 회로 정의
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_net(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (6, n_qubits)}

# 3. QNN + 고전 신경망 정의
class QNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quantum_net, weight_shapes)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        q_outs = []
        for xi in x:
            xi_flat = xi.view(-1)[:n_qubits] * np.pi  # 0~1 -> 0~π 스케일링
            q_out = self.q_layer(xi_flat)
            q_outs.append(q_out)
        q_outs = torch.stack(q_outs)
        return self.classifier(q_outs).squeeze(1)

# 4. 얼리스타핑 클래스
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.best_model = None

    def __call__(self, model, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = {k: v.cpu() for k, v in model.state_dict().items()}  # CPU 복사
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 5. 평가 함수
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

# 6. 학습 함수
def train():
    model = QNNModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=5)

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data, targets = data.to(device), targets.to(device).float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        acc = evaluate(model)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} | Test Acc={acc:.4f}")

        if early_stopper(model, acc):
            print("Early stopping triggered.")
            break

    model.load_state_dict(early_stopper.best_model)
    return model, early_stopper.best_acc

# 7. 실행 및 결과 저장
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, best_acc = train()

# 제출 파일 생성 (시간 기준)
model.eval()
all_preds = []
indices = []

with torch.no_grad():
    start_idx = 0
    for data, _ in test_loader:
        batch_size = data.size(0)
        data = data.to(device)
        outputs = model(data)
        preds = torch.round(torch.sigmoid(outputs))
        all_preds.extend(preds.cpu().numpy().astype(int))
        indices.extend(range(start_idx, start_idx + batch_size))
        start_idx += batch_size

submission_df = pd.DataFrame({'Id': indices, 'Prediction': all_preds})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'./submission_{timestamp}.csv'
submission_df.to_csv(save_path, index=False)
print(f"🎯 제출 파일 저장 완료: {save_path}")

print(f"✅ Test Accuracy (0 vs 6): {best_acc:.4f}")
