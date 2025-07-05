from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import numpy as np
import pandas as pd

import os

def get_deepest_last_file(root_folder):
    deepest_files = []

    # 모든 하위 폴더와 파일 탐색
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filenames:
            # 전체 경로로 파일 리스트 구성
            full_paths = [os.path.join(dirpath, f) for f in filenames]
            # 오름차순 정렬
            full_paths.sort()
            # 최하단 경로와 정렬된 파일 리스트 저장
            deepest_files.append((dirpath.count(os.sep), full_paths))

    if not deepest_files:
        return None  # 파일 없음

    # 가장 깊은 디렉토리 기준으로 정렬
    deepest_files.sort(reverse=True, key=lambda x: x[0])
    # 가장 깊은 디렉토리의 마지막 파일명 리턴
    return os.path.basename(deepest_files[0][1][-1])

# 1. 데이터
path = './_data/kaggle/bank/'
train_df = pd.read_csv(path+'train.csv', index_col=0)
test_df = pd.read_csv(path+'test.csv', index_col=0)
submission_df = pd.read_csv(path+'sample_submission.csv')

le = LabelEncoder()
train_df['Geography'] = le.fit_transform(train_df['Geography'])
test_df['Geography'] = le.transform(test_df['Geography'])
train_df['Gender'] = le.fit_transform(train_df['Gender'])
test_df['Gender'] = le.transform(test_df['Gender'])

train_df = train_df.drop(columns=['CustomerId','Surname'], axis=1)
test_df = test_df.drop(columns=['CustomerId','Surname'], axis=1)

x = train_df.drop(columns=['Exited'], axis=1)
y = train_df['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.1,
    random_state=100
)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 2. 모델 구성
path = './_save/keras28_mcp/08_bike/'
filename = get_deepest_last_file(path)
model = load_model(path+filename)

# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test, verbose=0)
loss = results[0]
acc = results[1]
print("불러온 모델 : ", filename)
print("Loss:", loss)
print("Accuracy :", acc)

'''

'''