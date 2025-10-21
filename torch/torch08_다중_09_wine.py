from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,
                                                    shuffle=True, random_state=5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

print('#######################')
print(x_train.dtype)    # torch.float64
print(x_train.shape, y_train.shape)
print(type(x_train))

model = nn.Sequential(
    nn.Linear(13,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.Linear(8,3),
).to(DEVICE)

criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy
optimizer = optim.SGD(model.parameters(), lr=0.02)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss=criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=1000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch: {} loss: {}'.format(epoch,loss))
print('####################')

def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y_pred,y)
    return loss2.item()
last_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', last_loss)
y_predict = model(x_test)
y_predict = torch.argmax(y_predict, dim=1)
y_predict = y_predict.detach().cpu().numpy()
y_test = y_test.detach().cpu().numpy()                              


acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

# loss:  0.03650752827525139
# acc:  0.9722222222222222