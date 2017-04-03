# Package MLEV for Python

## Motivation
The goal of this project is to implement the most recently published Maximum Likelihood Estimator for Variance (MLEV) for linear models in Python. MLEV estimates the variance of i.i.d error term, which could be used for model selection. In comparison to the well-known least squarred estimator for variance (LSEV), MLEV has the following two advantages:
1. In low dimensional datasets, MLEV is more accurate when compared to LSEV
2. In high dimensional datasets, MLEV still works well whereas LSEV is not defined
(Due to the limitation of mathematical expression in README.md file, please refer to README.ipynb for further details)
