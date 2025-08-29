import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_pred = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x_train)
x_std = torch.std(x_train)

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.02)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=10000

for epoch in range(1,epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch: {} loss: {}'.format(epoch,loss))
    
def evaluate(model,criterion,x_test,y_test):
    model.eval()

    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test,y_predict)

    return loss2.item()

loss2 = evaluate(model,criterion,x_test,y_test)
x_pred = (x_pred.to(DEVICE) - x_mean) / x_std
results = model(x_pred)

print('최종 loss: ',loss2) 
print('[12,13,14] 의 예측값: ', results.detach().cpu().numpy())