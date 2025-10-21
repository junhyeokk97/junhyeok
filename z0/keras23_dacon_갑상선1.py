import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 경로 설정 및 데이터 로드
path = 'C:/Study25/_data/dacon/갑상선/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(train_csv.head())
#              Age Gender Country Race Family_Background Radiation_History Iodine_Deficiency       Smoke Weight_Risk Diabetes  Nodule_Size  TSH_Result  T4_Result  T3_Result  Cancer
# ID
# TRAIN_00000   80      M     CHN  ASN          Positive           Exposed        Sufficient  Non-Smoker   Not Obese       No     0.650355    2.784735   6.744603   2.575820       1
# TRAIN_00001   37      M     NGA  ASN          Positive         Unexposed        Sufficient      Smoker       Obese       No     2.950430    0.911624   7.303305   2.505317       1
# TRAIN_00002   71      M     CHN  MDE          Positive         Unexposed        Sufficient  Non-Smoker   Not Obese      Yes     2.200023    0.717754  11.137459   2.381080       0
# TRAIN_00003   40      F     IND  HSP          Negative         Unexposed        Sufficient  Non-Smoker       Obese       No     3.370796    6.846380  10.175254   0.753023       0
# TRAIN_00004   53      F     CHN  CAU          Negative         Unexposed        Sufficient  Non-Smoker   Not Obese       No     4.230048    0.439519   7.194450   0.569356       1
print(train_csv.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 87159 entries, TRAIN_00000 to TRAIN_87158
# Data columns (total 15 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   Age                87159 non-null  int64
#  1   Gender             87159 non-null  object
#  2   Country            87159 non-null  object
#  3   Race               87159 non-null  object
#  4   Family_Background  87159 non-null  object
#  5   Radiation_History  87159 non-null  object
#  6   Iodine_Deficiency  87159 non-null  object
#  7   Smoke              87159 non-null  object
#  8   Weight_Risk        87159 non-null  object
#  9   Diabetes           87159 non-null  object
#  10  Nodule_Size        87159 non-null  float64
#  11  TSH_Result         87159 non-null  float64
#  12  T4_Result          87159 non-null  float64
#  13  T3_Result          87159 non-null  float64
#  14  Cancer             87159 non-null  int64
print(train_csv.isnull().sum())
# None
# Age                  0
# Gender               0
# Country              0
# Race                 0
# Family_Background    0
# Radiation_History    0
# Iodine_Deficiency    0
# Smoke                0
# Weight_Risk          0
# Diabetes             0
# Nodule_Size          0
# TSH_Result           0
# T4_Result            0
# T3_Result            0
# Cancer               0
print(train_csv.DESCR)