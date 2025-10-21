import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(1, 64, 10, 10)

##### 1. AdaptiveAvgPool2d #####
# gap = nn.AdaptiveAvgPool2d((1,1))
# x = gap(x)
# print(x.shape)  # torch.Size([1, 64, 1, 1])

##### 2. AdaptiveAvgPool2d #####
# x = nn.AdaptiveAvgPool2d((1,1))(x)
# print(x.shape)  # torch.Size([1, 64, 1, 1])

##### 3. F.adaptive_avg_pool2d #####
x = F.adaptive_avg_pool2d(x, (1,1))
print(x.shape)  # torch.Size([1, 64, 1, 1])


x = x.view(x.size(0), -1)
print(x.shape)  # torch.Size([1, 64])

