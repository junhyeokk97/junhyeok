import torch

# Python 버전 확인
print('Pytorch verison: ', torch.__version__)

# CUDA 여부
cuda_available = torch.cuda.is_available()
print('CUDA: ', cuda_available)

# 사용 가능 GPU 갯수
qpu_count = torch.cuda.device_count()
print('GPU 갯수: ', qpu_count)

if cuda_available:
    # 현재 사용중인 GPU
    current_device = torch.cuda.current_device()
    print('GPU ID: ',current_device)
    print('GPU: ', torch.cuda.get_device_name(current_device))
else:
    print('GPU X')

# CUDA 버전 확인
print('CUDA version: ', torch.version.cuda)

# CUDNN 버전 확인
cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
    print('cudnn version: ', cudnn_version)
else:
    print('cudnn X')

# Pytorch verison:  2.7.1+cu118
# CUDA:  True
# GPU 갯수:  1
# GPU ID:  0
# GPU:  NVIDIA GeForce RTX 3050
# CUDA version:  11.8
# cudnn version:  90100


import tensorflow as tf

# pytorch 버전 확인
print('텐서플로우 버전:', tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print('GPU 있음')
else:
    print('GPU 없음')
    
# CUDA 버전
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print('CUDA버전 :',cuda_version)

# CUDNN 버전
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print('cuDNN 버전:',cudnn_version) 