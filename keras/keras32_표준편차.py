import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[1,2,3,1],
                [4,5,6,2],
                [7,8,9,3],
                [10,11,12,114],
                [13,14,15,115]])
print(data.shape)   # (5, 4)

#1) 평균
means = np.mean(data, axis=0)
print('평균: ', means)  # [ 7.  8.  9. 47.]

#2) 모집단 분산 (n으로 나눈다.)
population_variances = np.var(data, axis=0)
print('모집단 분산: ', population_variances)    # [  18.   18.   18. 3038.]

#3) 표본 분산 (n-1로 나눈다.) / 데이터가 적을 땐 안 좋음.
variances = np.var(data, axis=0, ddof=1) # ddof : n-1로 나눈다.
print('표본 분산: ', variances) # [  22.5   22.5   22.5 3797.5]     / 표본끼리 나눠서 작업을 하면 

#4) 표본 표준 편차
stdl = np.std(data, axis=0, ddof=1)
print('표본 표준 편차: ', stdl)   # [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5) 모집단 표준 편차    / 일반적으로 표준 편차라고 하면 모집단 표준 편차다.
std2 = np.std(data, axis=0)
print('모집단 표준 편차: ', std2)   # [ 4.24264069  4.24264069  4.24264069 55.11805512]

#6) StandardScaler              / 평균에서 기존 데이터의 값을 뺀 후, 모집단 표준 편차로 나눈 값.
scaler = StandardScaler()     # / 평균 - 데이터 값 / 모집단 표준 편차.
scaled_data = scaler.fit_transform(data)
print('StandardScaler: ', scaled_data)
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]