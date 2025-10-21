from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

augment_size = 100 # size of augment

(x_tr,y_tr), (x_ts,y_ts) = fashion_mnist.load_data()
print(x_tr.shape) # (60000, 28, 28)
print(x_tr[0].shape) # (28, 28)
# plt.imshow(x_tr[0], cmap = 'gray')
# plt.show()

aaa = np.tile(x_tr[0].reshape(28*28), augment_size).reshape(-1, 28,28,1) # augment # of data to 'augment_size' times
print(aaa.shape)    # (100, 28, 28, 1)  > agment_size 만큼 x_train 1번째 이미지를 만들어낸다.

datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,    
    vertical_flip=True,      
    # width_shift_range=0.3,   
    # height_shift_range=0.23,  
    rotation_range=10,        
    # zoom_range=1.2,          
    # shear_range=0.7,         
    # fill_mode='nearest'
)

xy_data = datagen.flow(
    aaa,    # x datas
    np.zeros(augment_size), # y datas which is all zero // y데이터 생성, 전부 0으로 된 y값.
    batch_size = augment_size,
    shuffle = False, ).next()

x_data, y_data = xy_data

print(x_data.shape)
print(y_data.shape)

exit()
# print(xy_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001D810891790>
# print(type(xy_data)) # <class 'tuple'>
# print(len(xy_data)) # 2
# print(xy_data[0].shape) # (100, 28, 28, 1)
# print(xy_data[1].shape) # (100,)

plt.figure(figsize = (7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(x_data[i], cmap = 'gray')
    
plt.show()

