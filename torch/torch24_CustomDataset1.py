import torch
from torch.utils.data import Dataset, DataLoader

# 1. 커스텀 데이터셋
class MyDataset(Dataset):
    def __init__(self):
        self.x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.y = [0, 1, 0, 1, 0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

# 2. 인스턴스 생성
dataset = MyDataset()

# 3. DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 출력
for batch_idx, (xb, yb) in enumerate(loader):   # x_batch, y_batch 의 index와 값 출력
    print('============배치: ', batch_idx, '============')
    print('x: ', xb)
    print('y: ', yb)
# ============배치:  0 ============
# x:  tensor([[4.],
#         [1.]])
# y:  tensor([1, 0])
# ============배치:  1 ============
# x:  tensor([[3.],
#         [5.]])
# y:  tensor([0, 0])
# ============배치:  2 ============
# x:  tensor([[2.]])
# y:  tensor([1])