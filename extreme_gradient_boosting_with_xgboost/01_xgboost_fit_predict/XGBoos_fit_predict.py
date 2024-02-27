"""
 XGBoost: Fit/Predict
 --------------------
 Here, you'll be working with churn data. This dataset contains imaginary data from a ride-sharing
 app with user behaviors over their first month of app usage in a set of imaginary cities as well
 as whether they used the service 5 months after sign-up. It has been preloaded for you into
 a DataFrame called churn_data - explore it in the Shell!

 Your goal is to use the first month's worth of data to predict whether the app's users will remain
 users of the service at the 5 months mark. This is a typical setup for a churn prediction problem.
 To do this, you'll split the data into training and test sets, fit a small xgboost model on the
 training set, and evaluate its performance on the test set by computing its accuracy.

 pandas and numpy have been imported as pd and np, and train_test_split has been imported from sklearn.model_selection.
 Additionally, the arrays for the features and the target have been created as X and y.

 Import xgboost as xgb.
 - Create training and test sets such that 20% of the data is used for testing.
 - Use a random_state of 123.
 - Instantiate an XGBoostClassifier.
 - Specify n_estimators to be 10 estimators and an objective of 'binary:logistic'.
 - Fit the Extreme Gradient Boosting classifier to the training set.
 - Predict the labels of the test set
"""

import numpy as np
import xgboost as xgb
import pandas as pd

pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split

churn_data = pd.read_csv('churn_data.csv')

# Features
X = churn_data.drop('month_5_still_here', axis=1)

# Target variable
y = churn_data['month_5_still_here']

# Split into train and test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123
)

# Create an Extreme Gradient Boosting Classifier
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=10,
    seed=123
)

# Fit to the training set
xgb_classifier.fit(X_train, y_train)

# Predict the labels of the test set
predictions = xgb_classifier.predict(X_test)

# Compute the accuracy
# accuracy =
predictions_correct = np.sum(predictions == y_test)
test_set_size = y_test.shape[0]
accuracy = predictions_correct / test_set_size

print(f"Accuracy: {accuracy}")

print(
    f"""
    Our model has an accuracy of {accuracy * 100}%.
    In the following exercies we'll learn how to tune our Extreme Gradient Boosting models.
    """
)
