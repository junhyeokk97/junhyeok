import numpy as np
data = [1,2,3,4,5]

print(np.percentile(data, 25))  # 2.0

data = [1,2,3,4]
print(np.percentile(data, 25))  # 1.75

"""

rank = (n -1) * (q / 100)
     = (4- 1) * (25 / 100)
     = 3 * 0.25 = 0.75          # index 위치가 0.75
     
보간법
작은 값 = data 0번째 = 10
큰 값 = data 1번째 = 20

백분위값 = 작은 값 + (큰 값 - 작은 값) * rank
        = 10 + (20 - 10) * 0.75
        = 10 + 7.5 = 17.5
        
"""