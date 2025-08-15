import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

tanh = lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = np.arange(-5, 5, 0.1)

y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()