import torch
import torch.nn as nn

x = torch.randn(1, 64, 10, 10)

gap = nn.AdaptiveAvgPool2d((1,1))
x = gap(x)
print(x.shape)  # torch.Size([1, 64, 1, 1])

x = x.view(x.size(0), -1)
print(x.shape)  # torch.Size([1, 64])