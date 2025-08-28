import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = './_data/kaggle/netflix/'
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
        return x, y
    
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

y_predict = []
y_true = []
total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load('./_save/torch/netflix.pth', map_location=DEVICE))

    for x_test, y_test in train_loader:
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE))
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())
        
        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)
        
print(total_loss)   # tensor(62921.6445, device='cuda:0')

from sklearn.metrics import r2_score
print(type(y_predict))
print(type(y_true))

y_predict = np.concatenate(y_predict).flatten()
y_true = np.concatenate(y_true).flatten()


r2 = r2_score(y_true, y_predict)
print(r2, total_loss.item())