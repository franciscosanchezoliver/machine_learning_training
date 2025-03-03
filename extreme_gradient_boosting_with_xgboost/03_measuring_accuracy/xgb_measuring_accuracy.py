"""
Measuring accuracy
------------------
You'll now practice using XGBoost's learning API through its baked in cross-validation 
capabilities.
XGBoost gets its performance and efficiency gains by utilizing its own optimized 
data structure for datasets called a DMatrix.

In the previous exercise, the input datasets were converted into DMatrix data 
on the fly, but when you use the xgboost cv object, you have to first explicitly 
convert your data into a DMatrix. So, that's what you will do here before 
running cross-validation on churn_data.

To do:
- Create a DMatrix called churn_dmatrix from churn_data using xgb.DMatrix().
- Perform 3-fold cross-validation by calling xgb.cv().
    - dtrain is your churn_dmatrix
    - params is your parameter dictionary
    - nfold is the number of cross-validation folds (3)
    - num_boost_round is the number of trees we want to build (5)
    - metrics is the metric you want to compute (this will be "error", which we 
    will convert to an accuracy).
"""

import pandas as pd
import xgboost as xgb
import os

# Read the data
churn_data = pd.read_csv(
    os.getcwd()
    + "\\extreme_gradient_boosting_with_xgboost\\01_xgboost_fit_predict\\churn_data.csv"
)

# Select the features and target
X = churn_data.drop('month_5_still_here', axis=1).values
y = churn_data['month_5_still_here'].values

# As we want to use Cross Validation, we need to specify that we want to create
# an DMatrix object, which is an object that XGBoost uses. And we need
# to do this before doing cross validation.
churn_matrix = xgb.DMatrix(
    data=X,
    label=y,
)

# The parameters that we are going to use when doing the cross validation
params = {
    "objective": "reg:logistic",
    "max_depth": 3
}

# Create a 3-fold cross validation
cv_results = xgb.cv(
    dtrain=churn_matrix,# Our data in the format DMatrix
    params=params,      # Parameters for each tree
    nfold=3,            # Number of cross-validation folds
    num_boost_round=5,  # Number of trees we want to build
    metrics="error",    # The metric we want to compute, this is accuracy
    as_pandas=True,     # Retrieve the result as a pandas DataFrame
    seed=123
)

print("Cross Validation:")
print(cv_results.head())

# To tell an accuracy we can get the last test accuracy
test_errors = cv_results['test-error-mean']
test_errors_last_fold = test_errors.iloc[-1]
accuracy = 1 - test_errors_last_fold

print(f"\nAccuracy: {accuracy}")
