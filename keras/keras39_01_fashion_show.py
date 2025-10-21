import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))
print(pd.value_counts(y_test))
exit()
aaa = 3
print(y_train[aaa])

import matplotlib.pyplot as plt
plt.imshow(x_train, 'gray')
plt.show()