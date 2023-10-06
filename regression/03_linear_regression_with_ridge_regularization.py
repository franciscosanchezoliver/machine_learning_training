"""
Regularized regression: Ridge
-----------------------------
Ridge regression performs regularization by computing the squared values
of the model parameters multiplied by alpha and adding them to the loss function.

In this exercise, you will fit ridge regression models over a range of
different alpha values, and print their scores. You will use all the features
in the sales dataset to predict "sales".

Try different values for alpha and check which one gives the best performance.
"""

from sklearn.linear_model import Ridge
import pandas as pd

sales_df = pd.read_csv('advertising_and_sales_clean.csv')

# Select the features and target variables

print(sales_df)

# https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-6320c92e-31c3-48fb-9382-6a9169125722?ex=12 #
