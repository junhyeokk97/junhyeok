import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True,    # 수평 반전 > 데이터 증폭 또는 변환.
    vertical_flip=True,      # 수직 반전 > 데이터 증폭 또는 변환
    width_shift_range=0.1,   # 평행 이동 10% > 데이터 증폭 또는 변환
    height_shift_range=0.1,  # 수직 이동
    rotation_range=5,        # 5도 회전
    zoom_range=1.2,          # 1.2배 확대
    shear_range=0.7,         # 좌표 하나를 고정 시키고, 다른 몇 개의 좌표를 이동 시키는 변환
                             # 이미지를 늘려서 증폭 또는 변환.
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_path = './_data/image/brain/train/'
test_path = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    train_path,                 # 경로
    target_size=(200, 200),     # 리사이즈, 사이즈 규격 일치, 데이터가 크면 축소/ 작으면 확대
    batch_size=160,              # 이미지 파일을 배치 사이즈로 묶어서 훈련. / 160개의 이미지에 사이즈 10으로 진행 시, 16번의 훈련.
    class_mode='binary',        # 
    color_mode='grayscale',     # 
    shuffle=True,
)
    # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    test_path,                 # 경로
    target_size=(200, 200),     # 리사이즈, 사이즈 규격 일치, 데이터가 크면 축소/ 작으면 확대
    batch_size=10,              # 이미지 파일을 배치 사이즈로 묶어서 훈련. / 160개의 이미지에 사이즈 10으로 진행 시, 16번의 훈련.
    class_mode='binary',        # 
    color_mode='grayscale',     # 
    # shuffle=True,             # 평가는 셔플X / 셔플의 디폴트는 False
)
    # Found 120 images belonging to 2 classes.
    
print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002C979930130>
print(xy_train[0])
print(len(xy_train))    # xy_train 의 배치가 몇 개인지 확인
print(xy_train[0][0].shape) # (10, 200, 200, 1)
print(xy_train[0][1].shape) # (10,)
print(xy_train[0][0]) # 첫 번째 배치의 x 데이터
print(xy_train[0][1]) # 첫 번째 배치의 y 데이터
# x_train = xy_train[0][0]
# y_train = xy_train[0][1] 라고 생각하자.
exit()
# print(xy_train[0].shape)    # AttributeError: 'tuple' object has no attribute 'shape' tuple은 shape가 없다.
# print(xy_train[16])         # ValueError: Asked to retrieve element 16, but the Sequence has length 1 배치가 16개 이기 때문에 [15]까지
# print(xy_train[0][2])       # IndexError: tuple index out of range // x,y만 있기 때문에 [2]은 존재x [1]까지 존재

print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> // 1번째 배치의 x 데이터
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> // 1번째 배치의 y 데이터