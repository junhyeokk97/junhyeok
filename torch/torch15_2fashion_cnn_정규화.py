import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.datasets import FashionMNIST
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

#1. 데이터
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))]) # 규격을 동일하게 맞춘다. >> 56x56
# to.Tensor = 토치텐서 바꾸기 + minmaxScaler
######## tr.Normalize((0.5), (0.5)) ########
# Z_score Normalization (정규화의 표준화?)
# (x - 평균) / 표준편차
# (x - 0.5) / 0.5       위 식처럼 해야 하는데 통상 평균 0.5, 표준편차 0.5로 계산한다.
# -1 ~ 1 사이의 범위가 나오니 이미지 전처리에서는 통상 0.5, 0.5로 한다.
############################################

path = './_data/torch/'
train_dataset = FashionMNIST(path, train=True, download=True, tranform=transf)
test_dataset = FashionMNIST(path, train=False, download=True, tranform=transf)
print(len(train_dataset))   # 60000
print(train_dataset[0][1])  # 5

img_tensor, label = train_dataset[0]
print(label)    # 5
print(img_tensor.shape) # ([1, 56, 56]) 컬러, 가로, 세로
print(img_tensor.min(), img_tensor.max())   # tensor(0.) tensor(0.9922)


print(train_dataset)
print(type(train_dataset))  # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])     # (<PIL.Image.Image image mode=L size=28x28 at 0x274808B1E80>, 5)

x_train, y_train= train_dataset.data/255., train_dataset.targets
x_test, y_test= test_dataset.data/255., test_dataset.targets

print(x_train.shape, y_train.size())    # torch.Size([60000, 28, 28]) torch.Size([60000])
print(np.min(x_train.numpy()), np.max(x_train.numpy())) # 0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_train.reshape(-1, 784)
print(x_train.shape, x_test.size())

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
print(len(train_loader))    # 1875    >>> 60000/32


class CNN(nn.module):
    def __init__(self, num_features):
        # super().__init__()
        super(CNN, self).__init__

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),   # (1,56,56) >> (64,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),      # (n,64,27,27)
            nn.Dropout(0.2),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(num_features, 64, 32, kernel_size=(3,3), stride=1),   # (n,32,25,25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),    # (n,32,12,12)
            nn.Dropout(),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(num_features, 32, 16, kernel_size=(3,3), stride=1),   # (n,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),    # (n,16,5,5)
            nn.Dropout(),
        )
        self.flatten = nn.Flatten()
        self.hidden_layer4 = nn.Sequential(     # flatten (n,16,5,5)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        self.output_layer = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = CNN(1).to(DEVICE)    # torch CNN은 channel 값만 input으로 넣어준다.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()

        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(hypothesis, 1)
        acc = (y_pred == y_batch).float().mean()

        epoch_loss += loss.item()
        epoch_acc = acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

EPOCH = 100 
for epoch in range(1, EPOCH+1):
    loss, acc = train(model,criterion,optimizer, train_loader)
    print(f'epoch: {epoch}, loss: {loss:.4f}, accL {acc:.3f}')
    
def evaluate(model,criterion,loader):
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()

            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean()

            epoch_loss += loss.item()
            epoch_acc = acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)


EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch: {epoch}, loss: {loss:.4f}, acc: {acc:.3f}, \
          val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}')

loss, acc = evaluate(model, criterion, test_loader)
print('================================')
print('최종 loss: ', loss)
print('최종 acc: ', acc)



l_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', l_loss)
y_predict = model(x_test)
y_predict = np.round(y_predict.detach().cpu().numpy())
y_test = y_test.detach().cpu().numpy()                              

r2 = r2_score(y_test, y_predict)
print('acc: ', r2)