import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random

seed = 4131
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/Thyroid/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.isna().sum())   # 결측치 x
# print(test_csv.isna().sum())    # 결측치 x

train_csv['is_train'] = 1
test_csv['is_train'] = 0
# print(train_csv.shape, test_csv.shape)  # (87159, 16) (46204, 15)

combined = pd.concat([train_csv, test_csv], axis=0)
# print(combined)
#              Age Gender Country Race Family_Background Radiation_History Iodine_Deficiency       Smoke Weight_Risk Diabetes  Nodule_Size  TSH_Result  T4_Result  T3_Result  Cancer  is_train
# ID
# TRAIN_00000   80      M     CHN  ASN          Positive           Exposed        Sufficient  Non-Smoker   Not Obese       No     0.650355    2.784735   6.744603   2.575820     1.0         1
# TRAIN_00001   37      M     NGA  ASN          Positive         Unexposed        Sufficient      Smoker       Obese       No     2.950430    0.911624   7.303305   2.505317     1.0         1
# TRAIN_00002   71      M     CHN  MDE          Positive         Unexposed        Sufficient  Non-Smoker   Not Obese      Yes     2.200023    0.717754  11.137459   2.381080     0.0         1
# TRAIN_00003   40      F     IND  HSP          Negative         Unexposed        Sufficient  Non-Smoker       Obese       No     3.370796    6.846380  10.175254   0.753023     0.0         1
# TRAIN_00004   53      F     CHN  CAU          Negative         Unexposed        Sufficient  Non-Smoker   Not Obese       No     4.230048    0.439519   7.194450   0.569356     1.0         1
# ...          ...    ...     ...  ...               ...               ...               ...         ...         ...      ...          ...         ...        ...        ...     ...       ...
# TEST_46199    17      F     RUS  ASN          Negative           Exposed        Sufficient  Non-Smoker   Not Obese       No     0.050563    7.356242   7.729139   1.038126     NaN         0
# TEST_46200    37      M     JPN  MDE          Negative         Unexposed         Deficient  Non-Smoker   Not Obese       No     3.010883    3.981898   6.739967   2.252667     NaN         0
# TEST_46201    18      M     IND  CAU          Negative         Unexposed        Sufficient      Smoker   Not Obese       No     4.780738    3.142235  11.883107   1.044195     NaN         0
# TEST_46202    39      M     IND  AFR          Negative         Unexposed        Sufficient  Non-Smoker   Not Obese       No     0.420837    3.534950   8.294455   1.774779     NaN         0
# TEST_46203    68      F     NGA  HSP          Positive         Unexposed         Deficient      Smoker   Not Obese       No     2.690105    9.494194   6.666820   3.076600     NaN         0
# print(combined.shape)   # [133363 rows x 16 columns]

aaa = pd.get_dummies(combined, columns = ['Gender','Country','Race','Family_Background','Radiation_History',
                                          'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'],
                                            drop_first=True,
                                            dtype=int)

# print(aaa)
# print(aaa.shape)

drop_features = ['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']
aaa = aaa.drop(columns = drop_features)

train_csv = aaa[aaa['is_train'] == 1].drop(columns='is_train')
test_csv = aaa[aaa['is_train'] == 0].drop(columns='is_train')

# print(train_csv.shape, test_csv.shape)  # (87159, 21) (46204, 21)
# print(train_csv.columns)
# Index(['Cancer', 'Gender_M', 'Country_CHN', 'Country_DEU', 'Country_GBR',
#        'Country_IND', 'Country_JPN', 'Country_KOR', 'Country_NGA',
#        'Country_RUS', 'Country_USA', 'Race_ASN', 'Race_CAU', 'Race_HSP',
#        'Race_MDE', 'Family_Background_Positive', 'Radiation_History_Unexposed',
#        'Iodine_Deficiency_Sufficient', 'Smoke_Smoker', 'Weight_Risk_Obese',
#        'Diabetes_Yes'],
#       dtype='object')

test_csv = test_csv.drop(['Cancer'], axis=1)
# print(test_csv.shape)   # (46204, 20)

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.3,
                                                    random_state=seed,
                                                     stratify=y)


# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)


smote = SMOTE(random_state=1093)
x_train, y_train = smote.fit_resample(x_train, y_train)

es = xgb.callback.EarlyStopping(
    rounds = 45,
    metric_name = 'logloss',
    data_name = 'validation_0',
    save_best = True,
)

model = XGBClassifier(
    n_estimators=100000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.75,
    colsample_bytree=0.7,
    gamma=3,
    min_child_weight=5.9,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=seed,
    callbacks = [es]    
)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)], verbose=200)

path1 = './_data/dacon/Thyroid/'
# model.save_weights(path + 'w_24_11_save.h5')
model.save_model(path + 'xgb_30_17_model.json')

results = model.score(x_test, y_test)
print('SCORE: ', results)

proba = model.predict_proba(x_test)[:, 1]

# 최적 threshold 탐색
from sklearn.metrics import f1_score

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.001):
    y_pred = (proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold = {best_threshold:.2f} with F1 score = {best_f1:.4f}")

# thresholds = np.arange(0.4, 0.5, 0.)
# f1s = [f1_score(y_test, (proba > t).astype(int)) for t in thresholds]
# best_t = thresholds[np.argmax(f1s)]
# print("Best Threshold:", best_t)

# 최적 threshold로 예측 결과 만들기
y_predict = (proba > best_f1).astype(int)

# F1 score 계산
f1 = f1_score(y_test, y_predict)

# test_csv에도 동일한 threshold 적용
test_proba = model.predict_proba(test_csv)[:, 1]
y_submit = (test_proba > best_f1).astype(int)

# 제출 파일 생성
sub_csv['Cancer'] = y_submit
sub_csv.to_csv(path + 'sub_0630_17.csv', index=False)

print('f1 score: ', f1)

# SCORE:  0.8835773601398601      0630_1
# Best threshold = 0.53 with F1 score = 0.4864
# f1 score:  0.4839789574366332

# SCORE:  0.8875655594405595      0630_2 4124
# Best threshold = 0.52 with F1 score = 0.5084
# f1 score:  0.5080201101268853

# SCORE:  0.8779501748251748 0630_3
# Best threshold = 0.55 with F1 score = 0.4882
# f1 score:  0.4677272727272727

# SCORE:  0.8804632867132867      0630_4
# Best threshold = 0.69 with F1 score = 0.4729
# f1 score:  0.4586166471277843

# SCORE:  0.8815559440559441      0630_5
# Best threshold = 0.63 with F1 score = 0.4789
# f1 score:  0.4704212020823474

# SCORE:  0.882276649872799       0630_6
# Best threshold = 0.51 with F1 score = 0.4893
# f1 score:  0.4814340588988476

# SCORE:  0.8876141068489051      0630_7
# Best threshold = 0.51 with F1 score = 0.5101
# f1 score:  0.5093386068995825

# SCORE:  0.8830248915049633      0630_8
# Best threshold = 0.68 with F1 score = 0.4862
# f1 score:  0.47658104899633075\
    
# SCORE:  0.8867058654811415      0630_9
# Best threshold = 0.58 with F1 score = 0.5077
# f1 score:  0.5055590518145584

# SCORE:  0.8865534648921524      0630_10
# Best threshold = 0.53 with F1 score = 0.5053
# f1 score:  0.5033246020552086

# SCORE:  0.886369894446994       0630_11
# Best threshold = 0.52 with F1 score = 0.5052
# f1 score:  0.5035246727089627

# SCORE:  0.8876141068489051      0630_12
# Best threshold = 0.52 with F1 score = 0.5101
# f1 score:  0.5093386068995825

# SCORE:  0.8879134035017708      0630_13
# Best threshold = 0.51 with F1 score = 0.5102
# f1 score:  0.5096745822339489

# SCORE:  0.8861089047745124      0630_14
# Best threshold = 0.56 with F1 score = 0.5045
# f1 score:  0.5032007759456838

# SCORE:  0.8837533292358123      0630_15
# Best threshold = 0.50 with F1 score = 0.4880
# f1 score:  0.4833244159086856








# SCORE:  0.8821707205139973        0625_05
# f1 score:  0.4797365754812563

# SCORE:  0.8818838916934374      0625_06
# f1 score:  0.487173100871731

# SCORE:  0.8844079853143644        0625_07
# f1 score:  0.49000253100480895

# SCORE:  0.8867599816429554         0625_08
# f1 score:  0.5032712632108707

# SCORE:  0.8869894446994034      0625_09
# f1 score:  0.5097063215530114

# SCORE:  0.8871041762276274      0625_10
# f1 score:  0.5104477611940298

# SCORE:  0.8869894446994034      0625_11
# f1 score:  0.5101939333664842

# SCORE:  0.8873336392840753      0625_12
# f1 score:  0.5109561752988048

# SCORE:  0.8869894446994034     0625_14 186
# Best Threshold: 0.4999999999999998
# f1 score:  0.5101939333664842

# SCORE:  0.8867599816429554      0625_15
# Best Threshold: 0.5099999999999998
# f1 score:  0.5101939333664842

# SCORE:  0.8858421294171638
# Best Threshold: 0.45999999999999985
# f1 score:  0.4905079527963058

# SCORE:  0.8860715924736118      0626_3      0.5107
# Best Threshold: 0.47999999999999976
# f1 score:  0.4905079527963058

# SCORE:  0.8860715924736118      0626_4      0.5101
# Best Threshold: 0.48999999999999977
# f1 score:  0.4905079527963058

# SCORE:  0.8856126663607159      0626_6
# Best threshold = 0.46 with F1 score = 0.4908
# f1 score:  0.4876543209876544

# SCORE:  0.8887104176227627          0626_9
# Best threshold = 0.46 with F1 score = 0.4932
# f1 score:  0.49320794148380354

# SCORE:  0.8877629063097514          0626_10
# Best threshold = 0.45 with F1 score = 0.5188
# f1 score:  0.5184577522559475

# SCORE:  0.8897082923631596           0626_11
# Best threshold = 0.50 with F1 score = 0.5200
# f1 score:  0.5189421015010722

# SCORE:  0.8826009637448371          0626_11(2)
# Best threshold = 0.44 with F1 score = 0.4829
# f1 score:  0.48219941720511844

# SCORE:  0.8867026158788435          0626_12
# Best threshold = 0.48 with F1 score = 0.5044
# f1 score:  0.5027679919476598

# SCORE:  0.8865305185865076          0626_13
# Best threshold = 0.46 with F1 score = 0.5041
# f1 score:  0.5022647206844489

# SCORE:  0.8869894446994034        0626_15
# Best threshold = 0.49 with F1 score = 0.5102
# f1 score:  0.5089641434262948

# SCORE:  0.8869894446994034          0626_16
# Best threshold = 0.49 with F1 score = 0.5102
# f1 score:  0.5089641434262948

# SCORE:  0.8868747131711794          0626_17
# Best threshold = 0.46 with F1 score = 0.5099
# f1 score:  0.5097063215530114

# SCORE:  0.8871041762276274          0626_19
# Best threshold = 0.46 with F1 score = 0.4936
# f1 score:  0.49278350515463915

# SCORE:  0.8871041762276274          0626_20
# Best threshold = 0.46 with F1 score = 0.4936
# f1 score:  0.49278350515463915          

# SCORE:  0.8857273978889398          0626_21
# Best threshold = 0.51 with F1 score = 0.4938
# f1 score:  0.49056603773584906

# SCORE:  0.8871041762276274          0626_23
# Best threshold = 0.50 with F1 score = 0.4985
# f1 score:  0.4979591836734694

# SCORE:  0.8874370578112053      0626_23(2)
# Best threshold = 0.50 with F1 score = 0.4997
# f1 score:  0.498868778280543

# SCORE:  0.8866528744279355          0626_25
# Best threshold = 0.68 with F1 score = 0.4990
# f1 score:  0.4972191323692992

# SCORE:  0.8871041762276274          0626_27
# Best threshold = 0.50 with F1 score = 0.4994
# f1 score:  0.4988546703995928

# SCORE:  0.8858508604206501          0626_29
# Best threshold = 0.51 with F1 score = 0.4870
# f1 score:  0.4843549078439777