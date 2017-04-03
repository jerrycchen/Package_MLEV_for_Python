# Required python packages
import numpy as np
import pandas as pd

# dataset1 (n=100 >> p=10)
X1 = np.random.rand(100, 10)
beta1 = np.random.normal(loc=0, scale=(10/10)**0.5, size=10)
error1 = np.random.normal(loc=0, scale=10**0.5, size=100)
y1 = np.dot(X1, beta1) + error1
dataset1 = {'y': y1, 'X': X1}

# dataset2 (n=1000 << p=10000)
X2 = pd.DataFrame(np.random.randn(1000, 10000))
beta2 = np.random.normal(loc=0, scale=(10/1e4)**0.5, size=int(1e4))
error2 = np.random.normal(loc=0, scale=10**0.5, size=1000)
y2 = pd.Series(np.dot(X2, beta2) + error2)
dataset2 = {'y': y2, 'X': X2}
