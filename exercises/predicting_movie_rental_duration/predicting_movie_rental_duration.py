"""
In this project, you will use regression models to predict the number of days a 
customer rents DVDs for.

As with most data science projects, you will need to pre-process the data 
provided, in this case, a csv file called rental_info.csv. Specifically, you 
need to:

1. Read in the csv file rental_info.csv using pandas.

2. Create a column named "rental_length_days" using the columns "return_date" 
and "rental_date", and add it to the pandas DataFrame. This column should 
contain information on how many days a DVD has been rented by a customer.

3. Create two columns of dummy variables from "special_features", which takes 
the value of 1 when:
The value is "Deleted Scenes", storing as a column called "deleted_scenes".
The value is "Behind the Scenes", storing as a column called "behind_the_scenes".

4. Make a pandas DataFrame called X containing all the appropriate features you 
can use to run the regression models, avoiding columns that leak data about 
the target.
- Choose the "rental_length_days" as the target column and save it as a 
pandas Series called y.

5. Following the preprocessing you will need to:
- Split the data into X_train, y_train, X_test, and y_test train and test sets, 
avoiding any features that leak data about the target variable, and include 20% 
of the total data in the test set.
- Set random_state to 9 whenever you use a function/method involving 
randomness, for example, when doing a test-train split.
- Recommend a model yielding a mean squared error (MSE) less than 3 on the 
test set

Save the model you would recommend as a variable named best_model, and save its 
MSE on the test set as best_mse.
"""

import pandas as pd
import os
from movie_rental_utils import *
from sklearn.model_selection import train_test_split

# 1. Read in the csv file rental_info.csv using pandas.
df_rental = read_rental_data()

# 2. Create a column named "rental_length_days" using the columns "return_date"
# and "rental_date", and add it to the pandas DataFrame. This column should
# contain information on how many days a DVD has been rented by a customer.
df_rental = calculate_rental_length_days(df_rental)

# 3. Create two columns of dummy variables from "special_features", which takes
# the value of 1 when:
# The value is "Deleted Scenes", storing as a column called "deleted_scenes".
# The value is "Behind the Scenes", storing as a column called "behind_the_scenes".
df_rental = add_dummy_variables(df_rental)

# 4. Make a pandas DataFrame called X containing all the appropriate features you
# can use to run the regression models, avoiding columns that leak data about
# the target.
X = select_features(df_rental)
y = df_rental["rental_length_days"]

# 5. Split the data into X_train, y_train, X_test, and y_test train and test sets,
# avoiding any features that leak data about the target variable, and include 20%
# of the total data in the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_VALUE, random_state=RANDOM_STATE_VALUE
)

# Selection of important features for the model by using lasso
relevant_features = select_relevant_features(X_train, y_train)
X_train = X_train[relevant_features]
X_test = X_test[relevant_features]

# Now, we'll try several models, to find the best one. We'll add the results
# of each model to a dictionary, so we can compare the models later.
model_results_dict = {}


def add_model_results(model_name, model_result):
    model_results_dict[model_name] = model_result


# Linear Regression model
linear_regression_result = ols_linear_regression(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

add_model_results("Linear Regression", linear_regression_result)

# Random Forest Regression model
random_forest_regression_result = random_forest_regression(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

add_model_results("Random Forest Regression", random_forest_regression_result)

# Decision Tree Regressor
decision_tree_regressor_result = decision_tree_regressor(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

add_model_results("Decision Tree Regressor", decision_tree_regressor_result)


xgb_regressor_result = xgb_regresoor(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

add_model_results("XGB Regressor", xgb_regressor_result)


def get_best_model():
    min_model = None
    for _, model_data in model_results_dict.items():
        metric_name = "mse"
        metric_obtained = model_data[metric_name]

        # Set the min model as the first model compared
        if min_model is None:
            min_model = model_data

        # Check if this model has a lower metric than the min model
        if metric_obtained < min_model[metric_name]:
            min_model = model_data

    # Return the model with the min metric
    return min_model


best_model = get_best_model()

print("end")
