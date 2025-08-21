import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print('torch: ', torch.__version__ ,'사용 divece: ', DEVICE)

path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv (path + 'sample_submission.csv', index_col=0)

# print(train_csv)
# print(train_csv.head) #디폴트는 5개
# print(train_csv.tail)
# print(train_csv.head(10)) #이렇게 설정할 수 있다.

# print(train_csv.isna().sum())
# print(test_csv.isna().sum())
# print(train_csv.columns)
# #Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

#문자 데이터 수치화!!!
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
le_geo = LabelEncoder()
le_geo.fit(train_csv['Geography'])               
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv ['Geography'] = le_geo.transform(test_csv ['Geography'])

le_gen = LabelEncoder()
le_gen.fit(train_csv['Gender'])                  
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv ['Gender'] = le_gen.transform(test_csv ['Gender'])   


print(train_csv['Gender'].value_counts())

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv  = test_csv .drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1) # (165034, 10)
y = train_csv['Exited'] #(165034,)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,stratify=y,
                                                    shuffle=True, random_state=5)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print('#######################')
print(x_train.dtype)    # torch.float64
print(x_train.shape, y_train.shape) # (398, 30) (398, 1)
print(type(x_train))

model = nn.Sequential(
    nn.Linear(10,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,8),
    # nn.SiLU(),
    nn.Linear(8,1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.002)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis=model(x_train)
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
        loss2 = criterion(y, y_pred)
    return loss2.item()
last_loss = evaluate(model,criterion,x_test,y_test)
print('loss: ', last_loss)

y_predict = model(x_test).to(DEVICE)
y_predict2 = (y_predict > 0.5).float()
acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict2.detach().cpu().numpy())
print('acc: ', acc)

# loss:  19.662452697753906
# acc:  0.8633058652447891