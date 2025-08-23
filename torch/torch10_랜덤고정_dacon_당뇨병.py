import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import random
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print('torch: ', torch.__version__ ,'사용 divece: ', DEVICE)

seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']


print(x.shape ,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print('#######################')
print(x_train.dtype)    # torch.float64
print(x_train.shape, y_train.shape) # (398, 30) (398, 1)
print(type(x_train))

model = nn.Sequential(
    nn.Linear(8,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,8),
    # nn.ReLU(),
    nn.Linear(8,6),
    # nn.SiLU(),
    nn.Linear(6,1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.008)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis=model(x_train)
    loss=criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=1250
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epoch: {} loss: {}'.format(epoch,loss))
print('####################')

def evaluate(model,criterion,x,y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()
last_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', last_loss)

y_predict = model(x_test).to(DEVICE)
y_predict2 = (y_predict > 0.5).float()
acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict2.detach().cpu().numpy())
print('acc: ', acc)

# loss:  23.421831130981445
# acc:  0.7575757575757576