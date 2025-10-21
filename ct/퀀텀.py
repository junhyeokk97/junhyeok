import multiprocessing
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

# ------------------------------------------------------------------------------------
# 메인 실행 블록 선언: Windows에서 멀티프로세싱 안정성을 위해
# __name__ == '__main__' 블록이 최상단에 오는 것을 권장하는 경우도 있으나,
# 스크립트의 가독성을 위해 함수/클래스 정의 후 메인 로직을 배치합니다.
# ------------------------------------------------------------------------------------

##############################
# 0️⃣ 경로 설정
##############################
base_path = './Study25/_data/quantum/'
os.makedirs(base_path, exist_ok=True)

##############################
# 1️⃣ Seed 고정
##############################
SEED =190
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# ✨ 수정 제안
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 결정론적 알고리즘 사용을 비활성화하여 유연성 부여
    torch.backends.cudnn.deterministic = True
    # cuDNN이 가장 적합한 알고리즘을 찾도록 허용
    torch.backends.cudnn.benchmark = False
    
##############################
# 2️⃣ 디바이스 선언
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 5개의 큐비트를 사용하는 양자 디바이스 선언
dev = qml.device("default.qubit", wires=5)

##############################
# 3️⃣ QNN 회로 정의
##############################
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    num_qubits = 5
    layers = 2

    # 입력 데이터(inputs)와 가중치(weights)를 인코딩
    for l in range(layers):
        for i in range(num_qubits):
            # 입력 데이터는 배치 형태로 들어옴 (batch_size, num_inputs)
            # PennyLane이 자동으로 배치 차원을 처리해줌
            qml.RX(inputs[:, i % inputs.shape[1]], wires=i)
            qml.RY(weights[(l * num_qubits + i) % weights.shape[0]], wires=i)

        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[num_qubits - 1, 0])

    # RZ는 마지막에 한 번만 적용
    for i in range(num_qubits):
        qml.RZ(weights[(i + weights.shape[0] // 2) % weights.shape[0]], wires=i)
    
    # ✨ 개선 사항: 각 큐비트의 기댓값을 모두 측정하여 다중 출력 생성
    # [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))] 대신 아래 코드를 사용
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

##############################
# 4️⃣ 데이터 준비 (Dataset 정의)
##############################
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_full = datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform_train)
test_dataset_full = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform_test)

# 클래스 0, 6 데이터 필터링 및 레이블 변환 (0 -> 0, 6 -> 1)
indices_train = [i for i, (_, label) in enumerate(train_dataset_full) if label in [0, 6]]
train_dataset = Subset(train_dataset_full, indices_train)
for i in range(len(train_dataset.dataset.targets)):
    if train_dataset.dataset.targets[i] == 6:
        train_dataset.dataset.targets[i] = 1

indices_test = [i for i, (_, label) in enumerate(test_dataset_full) if label in [0, 6]]
test_subset = Subset(test_dataset_full, indices_test)
for i in range(len(test_subset.dataset.targets)):
    if test_subset.dataset.targets[i] == 6:
        test_subset.dataset.targets[i] = 1

##############################
# 5️⃣ 모델 정의
##############################
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 클래식 CNN 부분
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 5)  # ✨ 개선 사항: QNN 입력 큐비트 수와 맞춤 (5개)
        self.norm = nn.LayerNorm(5) 
        
        # 양자 QNN 부분
        self.q_params = nn.Parameter(torch.rand(30)) # 양자 회로 가중치
        
        # QNN 출력 이후 클래식 레이어
        # ✨ 개선 사항: QNN 출력이 5개이므로 입력 차원을 5로 변경
        self.fc2 = nn.Linear(5, 32)
        self.fc3 = nn.Linear(32, 2) # 최종 2개 클래스로 분류

    def forward(self, x):
        # 클래식 CNN 특징 추출
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.norm(x)
        
        # ✨ 개선 사항: 양자 회로에 배치 전체를 한 번에 전달하여 효율성 극대화
        q_out = quantum_circuit(x, self.q_params)
        # PennyLane 출력이 튜플일 수 있으므로 텐서로 변환
        q_out = torch.stack(list(q_out), dim=1).to(torch.float32)

        # 최종 분류
        x = self.fc2(q_out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# ------------------------------------------------------------------------------------
# 메인 실행 블록
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 멀티프로세싱 시작 방식 설정 (Windows 호환성)
    # multiprocessing.set_start_method('spawn', force=True)

    # 4️⃣-2. 데이터 로더 정의
    # num_workers는 환경에 맞게 조절 (0으로 설정 시 메인 프로세스에서 처리)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
    test_eval_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # 6️⃣ 규격 검사
    model_for_specs = HybridModel()
    model_for_specs.eval()
    # ✨ 개선 사항: QNN 입력 차원을 5로 변경
    dummy_q_inputs = torch.randn(1, 5) 
    dummy_q_weights = model_for_specs.q_params.data
    q_specs = qml.specs(quantum_circuit)(dummy_q_inputs, dummy_q_weights)
    assert q_specs["num_tape_wires"] <= 8
    assert q_specs['resources'].depth <= 30
    assert q_specs["num_trainable_params"] <= 60
    print("✅ QNN 규격 검사 통과")

    total_params = sum(p.numel() for p in model_for_specs.parameters() if p.requires_grad)
    assert total_params <= 50000
    print(f"✅ 학습 전체 파라미터 수 검사 통과: {total_params}")
    del model_for_specs
    

    # 7️⃣ 학습
    model = HybridModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    early_stopping_patience = 50
    epochs_no_improve = 0
    best_model_path = os.path.join(base_path, 'best_model_improved.pth')

    epochs = 1500
    TARGET_ACC = 95.5
    target_model_path = None
    early_stop_flag = False

    print(f"\n🔵 [학습 시작] 총 {epochs} epochs, 목표 정확도: {TARGET_ACC}%\n{'-'*45}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total * 100
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_acc)
        new_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch:03d}] Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}% (lr {new_lr:.6f})")

        # 베스트 모델 저장
        if avg_acc > best_acc:
            best_acc = avg_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  🔥 [Best] 갱신: {best_acc:.2f}%  [파일 저장: {best_model_path}]")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️  개선 없음 [{epochs_no_improve}/{early_stopping_patience}]")
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n🛑 Early stopping: 개선 없음 {early_stopping_patience}회 초과")
                early_stop_flag = True
                break

        # 🎯 목표 정확도 도달시 별도 저장 및 종료
        if avg_acc >= TARGET_ACC and target_model_path is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_model_path = os.path.join(base_path, f"model_{now}_targetacc_{TARGET_ACC:.2f}.pth")
            torch.save(model.state_dict(), target_model_path)
            print(f"\n🎯 [목표 달성] {TARGET_ACC}% → 모델 저장: {target_model_path}")
            print("    👉 학습 조기 종료\n" + "-"*45)
            early_stop_flag = True
            break

    print(f"\n{'='*50}\n[학습 종료]  Best train acc: {best_acc:.2f}%\n{'='*50}")

    # 8️⃣ 모델 최종 저장
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if target_model_path is not None:
        final_model_path = target_model_path
        print(f"\n🎯 [최종 모델] 목표 정확도 달성 모델이 저장됨: {final_model_path}")
    else:
        final_model_path = os.path.join(base_path, f'model_{now}_final_train_acc_{best_acc:.4f}_improved.pth')
        if os.path.exists(best_model_path):
            torch.save(torch.load(best_model_path), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        print(f"\n💾 [최종 모델] {final_model_path} 저장 완료")

    best_state_dict = torch.load(final_model_path, map_location='cpu')
    param_txt_path = os.path.splitext(final_model_path)[0] + '_params.txt'

    with open(param_txt_path, 'w') as f:
        print("\n\n🟢 [베스트 모델 파라미터 (요약)]")
        for name, param in best_state_dict.items():
            param_data = param.data.cpu().numpy() if hasattr(param, 'data') else param.cpu().numpy()
            # 파라미터가 길면 앞부분만 출력
            preview = np.array2string(param_data.flatten()[:10], separator=', ', threshold=10)
            shape_info = f"{param_data.shape}"
            # 터미널에 출력
            print(f"  {name:30} shape={shape_info} values={preview} ...")
            # 파일로 저장 (전부)
            f.write(f"{name} shape={shape_info}\n{param_data}\n\n")
    print(f"\n📦 [저장] 베스트 모델 파라미터 전체가 '{param_txt_path}' 파일로 저장됨\n")

    # 9️⃣ 추론 및 제출 생성
    model.load_state_dict(torch.load(final_model_path))
    model.eval()
    all_preds = []
    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())

    with torch.no_grad():
        for images, labels in test_eval_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct_eval += (pred == labels).sum().item()
            total_eval += labels.size(0)

    eval_acc = (correct_eval / total_eval) * 100 if total_eval > 0 else 0
    print(f"\n✅ 0/6 클래스 테스트 정확도: {eval_acc:.2f}%")

    final_submission_preds = [0 if p == 0 else 6 for p in all_preds]
    csv_filename = f"{base_path}y_pred_{now}_improved.csv"
    pd.DataFrame({"y_pred": final_submission_preds}).to_csv(csv_filename, index=False, header=False)
    print(f"✅ 결과 저장 완료: {csv_filename}")