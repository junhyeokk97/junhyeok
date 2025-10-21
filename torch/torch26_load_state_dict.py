import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = 'C:/Study25/_data/kaggle/netflix/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
print(train)                # [967 rows x 6 columns]
print(train.info())         # 결측치 없음
print(train.describe())     # 

# data = train.iloc[:,1:4]
# data['종가'] = train['Close']
# print(data)                 # [967 rows x 4 columns] - (N, 30, 3) 으로 바꿀거임

# hist = data.hist()
# plt.show()

from torch.utils.data import DataLoader, Dataset, TensorDataset
class Custom_Data(Dataset):
    def __init__(self, df, timesteps):
        self.train_csv = train
        self.x = self.train_csv.iloc[:, 1:4].values
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        # MinMaxScaler 적용
        
        self.y = self.train_csv['Close'].values
        self.timesteps = timesteps
    
    # (967, 3) -> (N, 30, 3)    
    def __len__(self):
        return len(self.y) - self.timesteps     # 행 - timesteps
    
    def __getitem__(self, idx):
        x = self.x[idx : idx+self.timesteps]    # x[idx : idx + timesteps]
        y = self.y[idx+self.timesteps]          # y[idx + timesteps]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
custom_data = Custom_Data(df=train, timesteps=30)

train_loader = DataLoader(custom_data, batch_size=32)

for batch_idx, (xb, yb) in enumerate(train_loader):   # DataLoader는 iterator형태
    print(f'\n================== {batch_idx}번 batch ==================')
    print(f'x : {xb.shape}\ny : {yb.shape}')

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=3,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            )   # (N, 30, 64)
        
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x)
        # x = torch.reshape(-1, 30*64)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
from tqdm import tqdm
# 3. 모델 인스턴스화, 손실 함수, 옵티마이저 정의
model = RNN().to(DEVICE)
criterion = nn.MSELoss() # 회귀 문제이므로 MSE 손실 함수 사용
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 옵티마이저 사용

# 학습 설정
epochs = 100 # 에포크 수 설정

# 4. 모델 학습
print("\n--- 모델 학습 시작 ---")
for epoch in range(epochs):
    model.train() # 모델을 학습 모드로 설정
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE).unsqueeze(1) # MSELoss를 위해 yb의 차원 맞추기 (N) -> (N, 1)

        optimizer.zero_grad() # 기울기 초기화
        output = model(xb) # 순전파
        loss = criterion(output, yb) # 손실 계산
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("--- 모델 학습 완료 ---")

model.eval() # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
predictions = []
actuals = []

with torch.no_grad(): 
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        output = model(xb)
        predictions.extend(output.squeeze().cpu().numpy()) # GPU에서 CPU로 이동 후 numpy 배열로 변환
        actuals.extend(yb.squeeze().cpu().numpy())


save = 'C:/_save/torch/'
torch.save(model.state_dict(), save + 'netflix.pth')