import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, 'device: ', DEVICE)

#1. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]).transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])



# x = torch.FloatTensor(x) # 텐서플로우에서는 알아서 바꿔 줬지만 토치는 직접 바꿔줘야 한다.
# print(x)
# print(x.shape) # torch.Size([3])
# print(x.size()) # torch.Size([3]) => 이것도 많이 씀 (shape랑 똑같음)

# 벡터형 데이터 안먹힘 최소 메트릭스 이상 형태로 바꿔줘야함

x = torch.FloatTensor(x).to(DEVICE) # unsqueeze => 차원 늘리기 (1) => 차원 늘리기 위치 , reshape 써도 됨  


y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(x.size(), y.size()) # torch.Size([3]) torch.Size([3, 1])

############ standardscaler ############
x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x - x_mean) / x_std

print('scale: ', x)


#2 모델 구성 
# model = Sequential()
# model.add(Dense(1,input_dim=1))  # 순서 => output, input
# 이거랑 같음
# model = nn.Linear(1,1).to(DEVICE) # 순서 => input,output # y = wx + b

model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

# 사실 y = xw + b 임 => 행렬 연산이라서 순서가 바뀌면 결과가 바뀜   


#3 컴파일 , 훈련

# model.compile(loss='mse',optimizer = 'adam')

criterion = nn.MSELoss() 
# optimizer = optim.Adam(model.parameters(),lr=0.002)
optimizer = optim.SGD(model.parameters(),lr=0.002) # 경사하강법

def train(model,criterion,optimizer,x,y):
    # model.train() # 디폴트 [훈련모드], drop out,batchnomal 적용
    optimizer.zero_grad() # 기울기 초기화.
                          # 각 배치마다 기울기를 초기화(0으로) 하여, 기울기 누적에 의한 문제 해결
                          # 텐서플로우에는 이런 문제가 없음 
    hypothesis = model(x) # 가설(predict) ,모델을 정의한다.  
    loss = criterion(hypothesis,y) # loss = mse() = 시그마 (y - hypothesis)^2/n                       
    ######여기까지 순전파####
    
    loss.backward() # 기울기(gradient) 값까지만 계산.
    optimizer.step() # 가중치 갱신
    
    return loss.item() # 토치를 우리가 볼수있는 수치형으로 바꿔줌
    
epochs = 1000

for epoch in range(1,epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('epochs: {} loss: {}'.format(epoch,loss))
    

print('==========================================')

#4. 평가, 예측

# loss = model.evaluate(x,y)
def evaluate(model,criterion,x,y):
    model.eval() # [평가 모드]이거 안쓰면 값이 다르게 나와버림 ,Drop out , Batchnomal 사용하지 않는다.
    
    # with torch.no_grad(): # 기울기(gradient) 갱신하지 않겠다.
    #     y_predict = model(x)
    #     loss2 = criterion(y,y_predict) # loss의 최종값
    
    # with문과 같은 의미
    torch.set_grad_enabled(False)
    y_predict = model(x)
    loss2 = criterion(y, y_predict)
    torch.set_grad_enabled(True)

    return loss2.item()


loss2 = evaluate(model,criterion,x,y)
x_pred = (torch.Tensor([[10, 1.3]]).to(DEVICE) - x_mean) / x_std
results = model(x_pred)
print('최종 loss: ',loss2) 

print('[10, 1.3] 의 예측값: ', results.item())

# x_pred = (torch.Tensor([[4]]).to(DEVICE) - x_mean) / x_std
# result = model(x_pred)
# print('예측값:',results)
# print('예측값:',results.item())

