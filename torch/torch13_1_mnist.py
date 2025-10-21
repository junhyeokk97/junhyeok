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
path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)
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

class DNN(nn.module):
    def __init__(self, num_features):
        # super().__init__()
        super(DNN, self).__init__

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.hidden1_layer1(x)
        x = self.hidden1_layer2(x)
        x = self.hidden1_layer3(x)
        x = self.hidden1_layer4(x)
        x = self.output_layer(x)
        return x

model = DNN(784).to(DEVICE)

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