import numpy as np
import pandas as pd

# TypeError: Design matrix (X) must be numpy.ndarray, pandas.Dataframe or pandas.Series and response variable (y) must be numpy.ndarray or pandas.Series
wrong_X = {'x1': [1,2,3,4], 'x2': [4,3,2,1]}
wrong_y = pd.DataFrame([1,2,3,4])
print(type(wrong_X), type(wrong_y)) # (<type 'dict'>, <class 'pandas.core.frame.DataFrame'>)
wrong_Data = MLEV(wrong_X, wrong_y)

# ValueError: Missing values are not allowed in design matrix (X)
wrong_X = pd.DataFrame(dataset1['X'])
wrong_X.iloc[0,0] = np.NaN
my_y = dataset1['y']
wrong_Data = MLEV(wrong_X, my_y)

# ValueError: X_Transpose is rank deficient, remove linearly dependent rows from design matrix (X)
wrong_X = np.array([[1,1,1,1], [2,2,2,2], [3,4,5,6], [7,8,9,10], [11,12,9,10]])
my_y = np.array([1,2,3,4,5])
wrong_Data = MLEV(wrong_X, my_y)



