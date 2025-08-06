import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01

def leaky_relu(x):
    # return np.maximum(alpha*x, x)
    return np.where (x>0, x, alpha*x)
leaky_relu = lambda x: np.where (x>0, x, alpha*x)

x = np.arange(-5, 5, 0.1)

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()