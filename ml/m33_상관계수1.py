import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


datasets = load_iris()
print(datasets.feature_names)

x = datasets['data']
y = datasets .target

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)
df['Target'] = y
print(df.corr())
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
# Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()