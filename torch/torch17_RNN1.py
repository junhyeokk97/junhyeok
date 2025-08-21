import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.datasets import MNIST
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import TensorDataset, DataLoader
import warnings
import random

############# 랜덤 고정 #############
SEED = 5
random.seed(SEED)       # python 랜덤 고정
np.random.seed(SEED)    # numpy 랜덤 고정
torch.manual_seed(SEED) # torch 고정
torch.cuda.manual_seed(SEED)    # torch cuda 시드 고정
####################################

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch: ', torch.__version__, '사용 device: ', DEVICE)

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (7, 3, 1)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7])

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

aaa = iter(train_loader)
bbb = next(aaa)
print(bbb)

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size = 1,    # feature 개수 / tensorflow에서는 input_dim
            hidden_size = 32,  # output_node 개수 / tensorflow에서는 unit
            # num_layers = 1,  # default / RNN 은닉층 레이어의 갯수
            batch_first=False,  # feature timesteps batch -->> batch feature timesteps // default = False
            # True일 땐, (N ,3, 1)  False 사용 시, (3, N, 1)
        )   # (N, 3, 32)    > timesteps 만큼 증가. timesteps * batch_size
        # self.rnn_layer1 = nn.RNN(1, 32, batch_firest=True)
        # self.fc1 = nn.Linear(3*32, 16)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, h0 = self.rnn_layer1(x)      # h0 = h는 hidden   >   RNN은 순환 신경망이므로 출력 값의 개수가 늘어난다.
        x = self.relu(x)

        # x = x.reshape(-1, 3*32)
        x = x[:, -1, :]     # 파라미터 연산량 감소
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = RNN().to(DEVICE)

# from torchsummary import summary
# summary(model, (3, 1))
#         Layer (type)               Output Shape         Param #
# ================================================================
#                RNN-1  [[-1, 3, 32], [-1, 2, 32]]               0
#               ReLU-2                [-1, 3, 32]               0
#             Linear-3                   [-1, 16]           1,552
#             Linear-4                    [-1, 8]             136
#               ReLU-5                    [-1, 8]               0
#             Linear-6                    [-1, 1]               9
# ================================================================
# Total params: 1,697
# Trainable params: 1,697
# Non-trainable params: 0

from torchinfo import summary
summary(model, (2, 3, 1))
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RNN                                      [2, 1]                    --
# ├─RNN: 1-1                               [2, 3, 32]                1,120
# ├─ReLU: 1-2                              [2, 3, 32]                --
# ├─Linear: 1-3                            [2, 16]                   1,552
# ├─Linear: 1-4                            [2, 8]                    136
# ├─ReLU: 1-5                              [2, 8]                    --
# ├─Linear: 1-6                            [2, 1]                    9
# ==========================================================================================
# Total params: 2,817
# Trainable params: 2,817
# Non-trainable params: 0

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, criterion, loader, return_preds=False):
    model.eval()
    total_loss = 0
    preds = []
    trues = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(x_batch).view(-1)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            preds.extend(y_pred.cpu().numpy().reshape(-1))
            trues.extend(y_batch.cpu().numpy().reshape(-1))

    y_pred_all = np.array(preds)
    y_true_all = np.array(trues)
    r2 = r2_score(y_true_all, y_pred_all)

    if return_preds:
        return total_loss / len(loader), r2, y_true_all, y_pred_all
    else:
        return total_loss / len(loader), r2

EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss = train(model, criterion, optimizer, train_loader)
        
#####################
