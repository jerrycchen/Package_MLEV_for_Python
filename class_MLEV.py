# Required python packages
import numpy as np
import pandas as pd
import scipy.optimize as optimx

class MLEV:
    """
    This package is used to compute variance estimates based on maximum likelihood estimates
    """

    def __init__(self, X=None, y=None):
        """
        __init__ function for Class MLEV
        :param X: design matrix (allowed data type: numpy array)
        :param y: response variable (allowed data type: numpy array)
        """
        if not ((isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)) and (isinstance(y, np.ndarray) or isinstance(y, pd.Series))):
            raise TypeError('Design matrix (X) must be numpy.ndarray, pandas.Dataframe or pandas.Series and response variable (y) must be numpy.ndarray or pandas.Series')
        elif not (X.ndim==1 or X.ndim==2):
            raise TypeError('The dimension of design matrix (X) must be 1 or 2')
        elif not (y.ndim==1 or (y.ndim==2 and y.shape[1]==1)):
            raise TypeError('The dimension of response variable (y) is not 1')
        elif X.shape[0] != y.shape[0]:
            raise TypeError('The number of rows in design matrix (X.shape[0]) does not match with y.shape[0]')
        else:
            self.X = np.asarray(X, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
            if np.sum(np.isnan(self.y)) != 0:
                raise ValueError('Missing values are not allowed in response variable (y)')
            elif np.sum(np.isnan(self.X)) != 0:
                raise ValueError('Missing values are not allowed in design matrix (X)')
            elif np.linalg.matrix_rank(self.X.T) < np.min(self.X.shape):
                raise ValueError('X_Transpose is rank deficient, remove linearly dependent rows from design matrix (X)')
            else:
                self.n = float(self.X.shape[0])
                self.p = float(self.X.shape[1])
                self.theta2_init = float(1)
                print('Initiation is now complete')

    def describe(self):
        print('The dimension of design matrix is (' + str(int(self.n)) + ', ' + str(int(self.p)) + ')')
        return(self)

    def eigen(self):
        self.XXt = np.dot(self.X, self.X.T)
        self.lbd, self.vec = np.linalg.eigh(self.XXt)
        self.yTilde = np.dot(self.vec.T, self.y)
        self.yTildeSq = self.yTilde**2
        return(self)

    def mlevObj(self, theta2):
        out1 = np.log(np.sum(self.yTildeSq/(theta2/self.p*self.lbd+1.0))) + 1.0/self.n*np.sum(np.log(theta2/self.p*self.lbd+1.0))
        return(out1)

    def getTheta2(self, theta2_init=5.0):
        self.theta2_est = optimx.fmin_l_bfgs_b(func=self.mlevObj, x0=np.array([theta2_init]), bounds=[(0, None)], approx_grad=True)
        self.theta2_hat = self.theta2_est[0]
        return(self)

    def getTheta1(self):
        self.theta1_hat = (1.0/self.n) * np.sum(self.yTildeSq/(self.theta2_hat/self.p*self.lbd+1.0))
        return(self.theta1_hat)

    def getMLEV(self):
        self.mlev_hat = self.eigen().getTheta2().getTheta1()
        return(self.mlev_hat)
