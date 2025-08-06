import numpy as np
import matplotlib.pyplot as plt

# def selu(x, alpha, lmbda):
#     return lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

selu = lambda x, alpha, lmbda : lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

x = np.arange(-5, 5, 0.1)

y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()