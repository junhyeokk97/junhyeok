import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
             [9,8,7,6,5,4,3,2,1,0]]).T

timesteps = 5



# print(a)  # (10, 2)
# [[ 1  9]
#  [ 2  8]
#  [ 3  7]
#  [ 4  6]
#  [ 5  5]
#  [ 6  4]
#  [ 7  3]
#  [ 8  2]
#  [ 9  1]
#  [10  0]]

def split(dataset, timesteps):
    x,y = [], []
    for i in range(len(dataset) - timesteps + 1):
        x_subset = dataset[i : (i+timesteps)]
        x.append(x_subset)
        y_subset = dataset[i : (i+timesteps+1)][1]
        y.append(y_subset[:1])
    return np.array(x), np.array(y)


x,y = split(a, timesteps=timesteps)


print(x.shape)
print(x)
print(y.shape)
print(y)