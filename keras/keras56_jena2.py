import pandas as pd

path = './_data/kaggle/jena/'
# # csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# test_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# sub_csv = pd.read_csv(path + 'submissionss.csv', index_col=0)

# # csv = csv.drop(['p (mbar)', 'T (degC)', 'Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)',
# #                 'wv (m/s)','max. wv (m/s)'], axis=1)

# # csv.to_csv( path + 'submissionss.csv')

# # test_csv = test_csv.drop(['wd (deg)'], axis=1)
# # test_csv.to_csv(path + 'test.csv')

# sub_csv['wd (deg)'] = 0
# print(sub_csv.head())

# sub_csv.to_csv( path + 'sample_submissionss.csv')



# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, SimpleRNN, GRU
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.preprocessing import RobustScaler
# # RMSE
# path = './_data/kaggle/jena/'
# csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# test_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# sub_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)


# csv = csv.replace(-9999, np.nan)
# csv = csv.fillna(csv.median())
# train_csv = csv.to_numpy()
# # print(csv.isna().sum())
# # print(csv.info())
# print(type(train_csv))

# timesteps = 288


# def split(train_csv, timesteps, stride=1):
#     aa = []
#     for i in range(0, len(train_csv) - timesteps + 1, stride):
#         subset = train_csv[i : (i+timesteps)].astype(np.float32)
#         aa.append(subset)
        
#     return np.array(aa)

# aa = split(train_csv, timesteps=timesteps, stride=10)
# # print(aa.shape)

# print(f"aa.shape: {aa.shape}")


# # print(aa.shape)    # (420408, 144, 14)

# x = aa[:-144, :-144, :-1]
# y = aa[:-144, 144:, -1]
# test = aa[-144:, -288:-144 ,:-1]
# # print(x)
# # print(y)
# print(x.shape)  # (420120, 144, 13)
# print(y.shape)  # (420120, 144)
# print(test.shape)  # (420120, 144)

# import pandas as pd

# # CSV 파일 불러오기
# path = './_data/kaggle/jena/sample_submission.csv'
# df = pd.read_csv(path, index_col=0)

# # 인덱스를 datetime 형식으로 변환
# df.index = pd.to_datetime(df.index)

# # 원하는 시간 구간으로 필터링
# filtered_df = df.loc['2016-12-31 00:10:00':'2017-01-01 00:00:00']

# # 결과 확인
# print(filtered_df.head())
# print(f"남은 데이터 행 수: {len(filtered_df)}")

# # CSV로 저장 (필요시)
# filtered_df.to_csv(path + 'submit.csv')





csv = pd.read_csv(path +'jena_climate_2009_2016.csv')

col = 'wd (deg)'

csv.loc[:145, col] = None

csv[col] = csv[col].dropna().reset_index(drop=True)
csv[col] = csv[col].reindex(csv.index)

csv.to_csv('data_cleaned.csv', index=False)