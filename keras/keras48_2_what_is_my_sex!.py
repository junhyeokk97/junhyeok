import numpy as np
import pandas as pd

###### preparing data

    ### load saved data

D_path = './_data/kaggle/men_women/men_women/'
x_tr = np.load(D_path + 'k46_gender_x_tr.npy')
y_tr = np.load(D_path + 'k46_gender_y_tr.npy')
x_ts = np.load(D_path + 'k46_gender_x_ts.npy')
y_ts = np.load(D_path + 'k46_gender_y_ts.npy')

###### model
    ### modeling
        # load
from tensorflow.keras.models import load_model

model = load_model(D_path + 'model.hdf5')

###### compile / fit
    ### compile
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
# hist = model.fit(x_tr, y_tr, epochs = 100, batch_size = 64, 
#                 validation_split = 0.1, callbacks = [])

    ### save

###### evaluate / predict
from sklearn.metrics import r2_score

me = np.load('./_data/image/me/keras47_me.npy')

loss = model.evaluate(x_ts, y_ts)
pred = model.predict(me)
# r2 = r2_score(y_ts, np.round(pred))
print('loss : ', loss[0])
print('acc : ', loss[1])
if pred==0:
    print('men')
else:
    print('wemen')

