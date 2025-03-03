"""
Cross-validation for R-squared
------------------------------
Cross-validation is a vital approach to evaluating a model. It maximizes the 
amount of data that is available to the model, as the model is not only trained 
but also tested on all the available data.

In this exercise, you will build a linear regression model, then use 6-fold 
cross-validation to assess its accuracy for predicting sales using social media 
advertising expenditure. You will display the individual score for each of the 
six-folds.

To do:
- Predict the sales using the investing in radio and social media.
- Use cross validation and a linear regression.
- Analyze the results obtain by the cross validation:
    - Calculate: mean, standard deviation and 95% confidence interval.
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np


sales_df = pd.read_csv("regression\\advertising_and_sales_clean.csv")

# Selection of features and target variables
features = ['radio', 'social_media']
target = 'sales'

X = sales_df[features]
y = sales_df[target]

number_of_folds = 6
folds = KFold(n_splits=number_of_folds,
              shuffle=True,  # shuffle the data before doing the folds
              random_state=5)

# Create a linear regression model
linear_regression = LinearRegression()

# Compute the cross validation score.
cross_val_scores = cross_val_score(linear_regression,
                                   X, y,
                                   cv=folds)

print(f"""
Features
{", ".join(features)}

Target
{target}

Model
Linear Regression

Cross Validation scores ({number_of_folds} folds)
{cross_val_scores}

    CV Statistics:
        - Mean : {np.mean(cross_val_scores)}
        - Standard deviation: {np.std(cross_val_scores)} 
        - 95% confidence interval: {np.quantile(cross_val_scores, [0.025, 0.975])}
""")
