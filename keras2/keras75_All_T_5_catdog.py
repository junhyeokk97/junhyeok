from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50, ResNet101
from tensorflow.keras.applications import ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications import Xception

model_list = [VGG16(), VGG19(), ResNet50(), ResNet50(), ResNet101(),
              ResNet101V2(), ResNet152(), ResNet152V2(), DenseNet121(), DenseNet169(), DenseNet201(),
              InceptionV3(), InceptionResNetV2(), MobileNet(), MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(),
              NASNetMobile(), NASNetLarge(), EfficientNetB0(), EfficientNetB1(), EfficientNetB2(), Xception()]

for model in model_list:
    model.trainable = False

    print("=========================")
    print("모델명: ", model.name)
    print("전체 가중치 갯수: ", len(model.weights))
    print("훈련 가능 갯수: ", len(model.trainable_weights))