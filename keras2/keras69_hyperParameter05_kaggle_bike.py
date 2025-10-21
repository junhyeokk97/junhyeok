import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#region(데이터 정보)
# print(train_csv.shape)              # (10886, 12)
# print(train_csv.columns)            
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],dtype='object')
# print(test_csv.shape)               # (6493, 9)
# print(test_csv.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'], dtype='object')
# print(sampleSubmission_csv.shape)   #(6493, 2)
# print(train_csv.isna().sum())       # NaN 없음
#endregion

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # [10886 rows x 8 columns]
y = train_csv[['casual']]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=333, test_size=0.1)

def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(8,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }
    
hyperparameters = create_hyperparameter()
    
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters,
                           cv=2, n_iter=5,
                           verbose=1)

es = EarlyStopping(monitor='val_loss', mode='auto',
                   verbose=1, restore_best_weights=True, patience=10)

rlr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                        patience=1, verbose=1,
                        factor=0.8)



str = time.time()
model.fit(x_train, y_train, epochs=1, callbacks=[es, rlr], validation_split=0.1)
end = time.time()
y_pred = model.predict(x_test)

print('최적의 매개변수: ', model.best_estimator_)

print('r2_score: ', r2_score(y_test, y_pred))
y_pred_best = model.best_estimator_.predict(x_test)
print('best_r2_score: ', r2_score(y_test, y_pred_best))
print('time : ', round(end-str))

# r2_score:  0.5057038523787811
# 3/3 [==============================] - 0s 499us/step
# best_r2_score:  0.5057038523787811
# time :  61