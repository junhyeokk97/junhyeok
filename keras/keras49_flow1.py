## keras47 copy

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


###### image amplifier

datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    # horizontal_flip=True,    # 수평 반전 > 데이터 증폭 또는 변환.
    # vertical_flip=True,      # 수직 반전 > 데이터 증폭 또는 변환
    width_shift_range=0.3,   # 평행 이동 10% > 데이터 증폭 또는 변환
    height_shift_range=0.23,  # 수직 이동
    rotation_range=10,        # 5도 회전
    zoom_range=1.2,          # 1.2배 확대
    shear_range=0.7,         # 좌표 하나를 고정 시키고, 다른 몇 개의 좌표를 이동 시키는 변환
    #                          # 이미지를 늘려서 증폭 또는 변환.
    fill_mode='nearest'
)
it = datagen.flow(arr,
    batch_size=1             # 이미지 파일을 배치 사이즈로 묶어서 훈련. / 160개의 이미지에 사이즈 10으로 진행 시, 16번의 훈련.
)
print('================================================')
print(it) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001BAC5D1CB20>
print('================================================')
# aaa = it.next() # python 2.0 syntax
# print(aaa.shape) # (1, 100, 100, 3)
# aaa = next(it) # recent python syntax

fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize = (5,5))
for i in range(2):
    for j in range(5):
        batch = next(it) # randomly do in the IDG
        print(batch.shape) # (1, 100, 100, 3)
        batch = batch.reshape(100,100,3)
    
        ax[i][j].imshow(batch)
        ax[i][j].axis('off')

plt.show()



