from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
import random
############# 랜덤 고정 #############
SEED = 5
random.seed(SEED)       # python 랜덤 고정
np.random.seed(SEED)    # numpy 랜덤 고정
torch.manual_seed(SEED) # torch 고정
torch.cuda.manual_seed(SEED)    # torch cuda 시드 고정
####################################

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=5,test_size=0.1,
                                                    shuffle=True, )

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# model = nn.Sequential(nn.Linear(8,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,1),
#                       ).to(DEVICE)

########################### torch 데이터셋 만들기 ###########################
from torch.utils.data import TensorDataset  # x, y 합치기
from torch.utils.data import DataLoader     # batch 정의

##### 1. x, y 합친다.
train_set = TensorDataset(x_train, y_train) # tuple 형태
test_set = TensorDataset(x_test, y_test)
print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000024652CDB400>
print(type(train_set))  # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))   # 512
print(train_set[0]) # (tensor([ 1.9244, -0.4054,  1.8413,  2.0688, -0.1404,  0.1076,  0.8802,  1.0983,
                    # -0.8484, -1.1479,  0.9280, -0.9884,  0.8169,  1.1567, -0.2583, -0.3193,
                    # -0.1196,  0.1369, -0.9298, -0.7813,  1.9693, -0.3680,  1.8671,  2.0225,
                    #  1.0612,  0.3797,  0.7609,  1.5208, -0.3394, -0.7226], device='cuda:0'),
                    # tensor([0.], device='cuda:0'))
print(train_set[0][0])  # 첫 번째 x
print(train_set[0][1])  # 첫 번째 y

##### 2. batch 정의
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
print(len(train_loader))    # 6
print(train_loader)         # <torch.utils.data.dataloader.DataLoader object at 0x0000019EF28A04F0>
# print(train_loader[0])    # 에러
# print(train_loader[0][0]) # 에러

##### iterator 데이터 확인
#1. for문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break       # 첫 번째 배치 출력.

for x_batch, y_batch in train_loader:
    print(x_batch)
    print(y_batch)
    break       # 첫 번째 배치 출력.
#2. next() 사용
bbb = iter(train_loader)
# aaa = bbb.next()        # 파이썬 버전 업 후 .nexxt() 없어짐
aaa = next(bbb)
print(aaa)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # super(Model, self).__init__()       # nn.Module에 있는 Model과 self 다 쓰겠다.
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()

# 정의 구현
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x

model = Model(30, 1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=0.02)

def train(model,criterion,optimizer,loader):
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss=criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    return loss.item()

epochs=1000
for epoch in range(1 + epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    print('epoch: {} loss: {}'.format(epoch,loss))
print('####################')
    
def evaluate(model,criterion,loader):
    model.eval()
    with torch.no_grad():
       y_predict = model(x_batch)
       loss2 = criterion(y_predict, y_batch)
       total_loss = loss2.item()
    return total_loss / len(loader)
l_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', l_loss)
y_predict = model(x_test)
y_predict = np.round(y_predict.detach().cpu().numpy())
y_test = y_test.detach().cpu().numpy()                              


r2 = r2_score(y_test, y_predict)
print('acc: ', r2)

# loss:  0.42187660932540894
# acc:  0.6918208003044128