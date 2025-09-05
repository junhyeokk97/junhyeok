import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x = np.array(range(100))
y = np.array(range(1,101))
x_pre = np.array([101,102])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pre = torch.tensor(x_pre, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x_train)
x_std = torch.std(x_train)
# x_pred = (x-x_mean) / x_std

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.005)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=1000

for epoch in range(1,epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch: {} loss: {}'.format(epoch,loss))
    
def evaluate(model,criterion,x_test,y_test):
    model.eval()

    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_predict, y_test)

    return loss2.item()

loss2 = evaluate(model,criterion,x_test,y_test)
x_pred = (x_pre.to(DEVICE) - x_mean) / x_std
results = model(x_pred)

print('최종 loss: ',loss2) 
print('[12,13,14] 의 예측값: ', results.detach().cpu().numpy())