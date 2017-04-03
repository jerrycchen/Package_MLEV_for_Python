import numpy as np
import pandas as pd

# Test-1 regular numpy.ndarray datatype input, low dimension
my_X = dataset1['X']
my_y = dataset1['y']
print(type(my_X), type(my_y)) # (<type 'numpy.ndarray'>, <type 'numpy.ndarray'>)
my_Data = MLEV(my_X, my_y) # Initiation is now complete
my_Data.getMLEV() # 9.4361125899406293

# Test-2 regular pandas.DataFrame and pandas.Series datatype inputs, high dimension
my_X = dataset2['X']
my_y = dataset2['y']
print(type(my_X), type(my_y)) # (<class 'pandas.core.frame.DataFrame'>, <class 'pandas.core.series.Series'>)
my_Data = MLEV(my_X, my_y) # Initiation is now complete
my_Data.getMLEV() # 10.6155546083739




