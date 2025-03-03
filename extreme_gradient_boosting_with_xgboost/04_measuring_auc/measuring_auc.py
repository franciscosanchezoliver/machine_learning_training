"""
Measuring AUC
-------------
Now that you've used cross-validation to compute average out-of-sample accuracy 
(after converting from an error), it's very easy to compute any other metric 
you might be interested in. All you have to do is pass it (or a list of metrics) 
in as an argument to the metrics parameter of xgb.cv().

Your job in this exercise is to compute another common metric used in binary 
classification - the area under the curve ("auc"). As before, churn_data is 
available in your workspace, along with the DMatrix churn_dmatrix and parameter 
dictionary params.

To do:
- Perform 3-fold cross-validation with 5 boosting rounds and "auc" as your metric.
- Print the "test-auc-mean" column of cv_results.
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
X = churn_data.drop("month_5_still_here", axis=1).values
y = churn_data["month_5_still_here"].values

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

# Create a 3-fold cross validation but using "auc" as the metric instead 
# of using "error" as the metric
cv_results = xgb.cv(
    dtrain=churn_matrix,  # Our data in the format DMatrix
    params=params,  # Parameters for each tree
    nfold=3,  # Number of cross-validation folds
    num_boost_round=5,  # Number of trees we want to build
    metrics="auc",  # The metric we want to compute, this is accuracy
    as_pandas=True,
    seed=123
)

print("Cross Validation:")
print(cv_results.head())

print((cv_results["test-auc-mean"]).iloc[-1])
