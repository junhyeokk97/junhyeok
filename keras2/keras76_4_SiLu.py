# 2017 swish 구글 발표
import numpy as np
import matplotlib.pyplot as plt

def silu(x):
    return x * ( 1 / (1 + np.exp(-x)))

silu = lambda x : x * ( 1 / (1 + np.exp(-x)))

x = np.arange(-5, 5, 0.1)

y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()