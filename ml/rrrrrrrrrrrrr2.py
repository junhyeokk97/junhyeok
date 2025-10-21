import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision.transforms import ToTensor
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from torchvision import transforms

# ----------------------- 1. 데이터 준비 -----------------------
# 이미지 크기 2x2로 축소 (4픽셀 -> 2 큐비트)
transform = transforms.Compose([
    transforms.Resize((2, 2)), # 이미지 크기 2x2로 조절 (4픽셀)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_full = torchvision.datasets.FashionMNIST('./', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST('./', train=False, download=True, transform=transform)

# 0과 6만 추출
train_mask = (train_dataset_full.targets == 0) | (train_dataset_full.targets == 6)
train_indices = torch.where(train_mask)[0]
train_dataset_full.targets[train_dataset_full.targets == 6] = 1 # 6을 1로 매핑
binary_train_dataset = Subset(train_dataset_full, train_indices)

# Train/Val Split
train_len = int(0.8 * len(binary_train_dataset))
val_len = len(binary_train_dataset) - train_len
train_ds, val_ds = random_split(binary_train_dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------- 2. QNN 정의 -----------------------
n_qubits = 2 # 2x2 이미지 -> 4픽셀 -> 2^2 = 4차원 입력 가능
depth = 3 # QNN 깊이 정의. 너무 깊지 않게 시작
# n_params = 2 * n_qubits * depth # 이 변수는 이제 직접 사용하지 않음

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_net(inputs, weights):
    # AmplitudeEmbedding은 입력 벡터의 크기가 2^n_qubits여야 함 (여기서는 2^2=4)
    # 이미 2x2 이미지로 전처리했으므로 4픽셀 (1차원 벡터) 입력
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for d in range(depth):
        # 파라미터는 (depth, 2 * n_qubits) 형태로 전달되므로 인덱싱 조정
        for i in range(n_qubits):
            qml.RX(weights[d, i], wires=i)
            qml.RY(weights[d, i + n_qubits], wires=i)
        # CNOT 게이트 추가 (옵션: 연결성 강화)
        qml.CNOT(wires=[0, 1]) # n_qubits-1 대신 고정 CNOT 또는 패턴 변경
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] # n_qubits 만큼의 기댓값 출력

class QNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # QNN의 학습 가능한 파라미터
        self.q_params = nn.Parameter(torch.randn((depth, 2 * n_qubits), dtype=torch.float32))
        # QNN 출력 후 분류를 위한 선형 레이어 추가
        self.post_qnn_fc = nn.Linear(n_qubits, 2) # n_qubits 출력을 2 클래스로 매핑

    def forward(self, x_batch):
        batch_outputs = []
        for x in x_batch:
            # 입력 이미지를 1차원 벡터로 펼치고 n_qubits에 맞는 차원으로 자르기
            # 이미 2x2로 리사이즈했으므로 x는 1x2x2 -> 4차원
            x_flat = x.view(-1) # 4차원
            
            # AmplitudeEmbedding을 위한 정규화
            norm = torch.norm(x_flat)
            if norm == 0: # 0으로 나누는 것을 방지
                x_q = x_flat # 0벡터는 0으로 유지
            else:
                x_q = x_flat / norm
            
            # QNN 통과
            q_out = quantum_net(x_q, self.q_params)
            q_out = torch.stack(q_out) # (n_qubits,) 텐서로 변환
            
            # QNN 출력 후 고전적인 선형 레이어 통과
            logits = self.post_qnn_fc(q_out)
            
            # 최종 로짓에 로그 소프트맥스 적용
            batch_outputs.append(F.log_softmax(logits, dim=-1)) # dim=-1은 마지막 차원에 적용

        return torch.stack(batch_outputs)

# ----------------------- 3. 학습 (EarlyStopping 포함) -----------------------
model = QNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 학습률 조정 가능
loss_fn = nn.NLLLoss()

patience = 7 # 인내심 증가 (과적합 완화)
best_val_loss = float('inf')
wait = 0
best_model_state = None

for epoch in range(50): # 에포크 수 조정 가능
    model.train()
    train_loss = 0
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in bar:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"✅ Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss = loss_fn(output, target)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"📉 Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹ EarlyStopping triggered at epoch {epoch+1}")
            break

# Best 모델 불러오기
if best_model_state:
    model.load_state_dict(best_model_state)

# ----------------------- 4. 평가 -----------------------
model.eval()
y_true, y_pred = [], []

# test_dataset은 전체 FashionMNIST이므로, 0과 6에 해당하는 레이블만 필터링 필요
test_mask = (test_dataset.targets == 0) | (test_dataset.targets == 6)
test_indices = torch.where(test_mask)[0]
test_binary_dataset = Subset(test_dataset, test_indices)
test_binary_loader = DataLoader(test_binary_dataset, batch_size=32, shuffle=False)


with torch.no_grad():
    for data, target in tqdm(test_binary_loader, desc="Evaluating"): # 수정: test_binary_loader 사용
        output = model(data)
        preds = output.argmax(dim=1)
        y_pred.extend(preds.tolist())
        # 테스트셋의 6은 그대로 6으로, 0은 0으로 유지
        # 원래 0과 6만 추출했으므로, 6을 1로 매핑했던 훈련 데이터와 달리,
        # 테스트 데이터는 원래 레이블을 사용하거나, 훈련 데이터와 동일하게 6을 1로 매핑해야 함.
        # 여기서는 평가 시점에 6을 1로 매핑하는 것이 편의상 좋음.
        target_mapped = torch.where(target == 6, 1, target)
        y_true.extend(target_mapped.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
# 이미 0과 6만 있는 데이터셋에 대해 평가하므로 추가 마스킹 필요 없음
acc = (y_pred == y_true).mean() # 직접 비교
print(f"🎯 Accuracy (labels 0/6 only): {acc:.4f}")

# ----------------------- 5. 결과 저장 -----------------------
# y_pred_mapped = np.where(np.array(y_pred) == 1, 6, 0) # 1 -> 6, 0은 그대로 (훈련시 6을 1로 매핑했으므로 다시 원래대로)
# Note: 모델 예측은 0과 1로 나오므로, 제출 파일에는 원래 레이블 (0과 6)로 다시 매핑하여 저장.
y_pred_for_save = np.where(np.array(y_pred) == 1, 6, 0) # 1은 6으로, 0은 0으로

now = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"qnn_predictions_{now}.csv"
np.savetxt(filename, y_pred_for_save, fmt="%d")
print(f"📁 예측 결과 저장 완료: {filename}")