from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMbile
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import Xception

model_list = [VGG16(include_top=False,input_shape=(32, 32, 3)),
              ResNet50(include_top=False,input_shape=(32, 32, 3)),
              ResNet101(include_top=False,input_shape=(32, 32, 3)),
              DenseNet121(include_top=False,input_shape=(32, 32, 3)),
              InceptionV3(include_top=False,input_shape=(32, 32, 3)),
              InceptionResNetV2(include_top=False,input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False,input_shape=(32, 32, 3)),
              EfficientNetB0(include_top=False,input_shape=(32, 32, 3)),
              Xception(include_top=False,input_shape=(32, 32, 3))]

for model in model_list:
    model.trainable = False

    print("=========================")
    print("모델명: ", model.name)
    print("전체 가중치 갯수: ", len(model.weights))
    print("훈련 가능 갯수: ", len(model.trainable_weights))