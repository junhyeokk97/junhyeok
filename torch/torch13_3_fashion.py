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

SEED = 5
random.seed(SEED)       # python 랜덤 고정
np.random.seed(SEED)    # numpy 랜덤 고정
torch.manual_seed(SEED) # torch 고정
torch.cuda.manual_seed(SEED)  

path = './_data/torch/'
train_dataset = FashionMNIST(path, train=True, download=True)
test_dataset = FashionMNIST(path, train=False, download=True)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets


x_train = x_train.view(-1, 28*28)
x_test = x_test.view(-1, 28*28)
print(x_train.shape, x_test.size())



train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

class DNN(nn.Module):
    def __init__(self, num_features):
        super(DNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = DNN(784).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.02)

def train(model, criterion, optmizer, loader):
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
    
EPOCH = 10
for epoch in range(1, EPOCH+1):
    loss, acc = train(model,criterion,optimizer,train_loader)
    print(f'epoch: {epoch}, loss: {loss:.4f}, accL {acc:.3f}')
    
def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)

            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)
    
EPOCH = 10
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch: {epoch}, loss: {loss:.4f}, acc: {acc:.3f}, \
          val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}')
    
loss, acc = evaluate(model, criterion, test_loader)
print('================================')
print('최종 loss: ', loss)
print('최종 acc: ', acc)



l_loss = evaluate(model,criterion,test_loader)
print('loss: ', l_loss)
y_predict = model(x_test.to(DEVICE))
y_predict = torch.argmax(y_predict, dim=1).cpu().numpy()
y_test = y_test.detach().cpu().numpy()                              


acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

# 최종 loss:  1.6446196674919737
# 최종 acc:  0.3131988817891374
# loss:  (1.6444201339928868, 0.3133985623003195)
# acc:  0.313