# 선형을 비선형으로 만들 때 효과적

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)


pf = PolynomialFeatures(degree=3, include_bias=False)       # default : True
x_pf = pf.fit_transform(x)
print(x_pf)
#  include_bias=False
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

# include_bias=True
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

# degree=3
# [[  0.     1.     0.     0.     1.      0.       0.         0.          1.]
#  [  2.     3.     4.     6.     9.      8.       12.       18.         27.]
#  [  4.     5.    16.     20.    25.     64.      80.      100.        125.]
#  [  6.     7.    36.     42.    49.    216.      252.     294.        343.]]
#    x1     x2    x1^2    x1*x2   x2^2   x1^3    x1^2*x2   x1*x2^2     x2^3 

### 통상적으로
# 선형모델(lr등)에 쓸 경우에는 include_bias=True를 써서 1만 있는 컬럼을 만드는 게 좋음
# 왜냐하면 y = wx + b 의 bias = 1 의 역할을 하기 때문
# 비선형모델 (rf, xgb 등)에 쓸 경우에는 include_biae = False가 좋음