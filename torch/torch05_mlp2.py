import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]).T
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x = torch.FloatTensor(x).to(DEVICE) 
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)              # 위와 동일한 기능 / 권장사항
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x-x_mean) / x_std

model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000

for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('eopchs: {} loss: {}', format(epoch,loss))


def evaluate(model,criterion,x,y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    
    return loss2.item()

loss2 = evaluate(model, criterion,x,y)
x_pred = (torch.Tensor([[10, 1.9, 1]]).to(DEVICE) - x_mean) / x_std
results = model(x_pred)

print('최종 loss: ',loss2) 

print('[10, 1.3] 의 예측값: ', results.item())