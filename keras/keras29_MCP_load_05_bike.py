from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import os

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

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
path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv', index_col=0)
test_csv =  pd.read_csv(path + '/test.csv', index_col=0)
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)

# 2. 모델 구성
path = './_save/keras28_mcp/05_bike/'
filename = get_deepest_last_file(path)
model = load_model(path+filename)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# 4. 평가, 예측
results = model.predict(x_test)
loss = model.evaluate(x_test, y_test, verbose=0)
rmse = RMSE(results, y_test)
print("불러온 모델 : ", filename)
print("Loss:", loss)
print("RMSE :", rmse)

'''
Loss: 21361.49609375
RMSE : 146.15572549082708
'''