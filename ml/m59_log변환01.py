import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)

print(data)
print(data.shape)   # (1000,)
print(np.min(data), np.max(data))
# 0.0017566428571336938 20.559239747987064
log_data = np.log1p(data)   # np.expmlp(data)
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()
# np.log(data)