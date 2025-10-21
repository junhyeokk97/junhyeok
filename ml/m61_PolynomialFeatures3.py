# 선형을 비선형으로 만들 때 효과적

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)       # default : True
x_pf = pf.fit_transform(x)                          # interaction_only는 제곱을 빼버리는 느낌으로 성능이 잘 안 나온다고 본다.
print(x_pf)

# interaction_only=True
# [[  0.   1.   2.   0.   0.   2.]
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]

# interaction_only=False
# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]