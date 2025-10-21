from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import accuracy_score, r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=5,
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

model = nn.Sequential(nn.Linear(8,10),
                      nn.Linear(10,10),
                      nn.Linear(10,10),
                      nn.Linear(10,10),
                      nn.Linear(10,1),
                      ).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.02)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss=criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=1000
for epoch in range(1 + epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch: {} loss: {}'.format(epoch,loss))
print('####################')
    
def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
       y_predict = model(x)
       loss2 = criterion(y_predict, y)
    return loss2.item()
l_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', l_loss)
y_predict = model(x_test)
y_predict = y_predict.detach().cpu().numpy()
y_test = y_test.detach().cpu().numpy()                              


r2 = r2_score(y_test, y_predict)
print('acc: ', r2)