import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_breast_cancer
import random
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

seed = 50
random.seed(seed)
np.random.seed(seed)

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

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)


scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# model = nn.Sequential(nn.Linear(8,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,10),
#                       nn.Linear(10,1),
#                       ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # super(Model, self).__init__()       # nn.Module에 있는 Model과 self 다 쓰겠다.
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()
        

        
    def forward(self, x):       # 정의 구현
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x

model = Model(10, 1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.005)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss=criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=300
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


y_predict = (y_predict > 0.5).astype(int)

acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

# loss:  0.3256860673427582
# acc:  0.8613669413475521