from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt
import numpy as np

path = 'c:/before/study25/_data/image/me/'
img = load_img(path + 'today.jpg', target_size = (100,100),) 
print(img) # <PIL.Image.Image image mode=RGB size=100x100 at 0x2131CE8CE20>
print(type(img)) # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr.shape) # (100, 100, 3)
print(type(arr)) # <class 'numpy.ndarray'>

###### 3D -> 4D data
# arr = arr.reshape # (1,100,100,3)
# print(arr.shape) # (1, 100, 100, 3)

arr = np.expand_dims(arr, axis = 0)
print(arr.shape) # (1, 100, 100, 3)

np.save(path + 'keras47_me.npy', arr = arr)

