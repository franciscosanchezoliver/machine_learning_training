"""
Fit and predict for regression
------------------------------

Now you have seen how linear regression works, your task is to create a multiple
linear regression model using all the features in the sales dataset.

Dataset example:
     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83

You will then use this model to predict sales based on the values of the test features.

LinearRegression and train_test_split have been preloaded for you from their respective modules.

"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

sales_df = pd.read_csv('./advertising_and_sales_clean.csv')

sales_df.head()

# From all the features available, we are going to discord the column 'influencer'
# as it is a cathegorical variable.
X = sales_df[['tv', 'radio', 'social_media']]

# We want to predict how money are we going to make, this is the 'sales' column.
y = sales_df['sales']

# Separate the data into a train and a test set.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Create the linear regression model.
linear_regression_model = LinearRegression()

# Train the model to the training data.
linear_regression_model.fit(X_train, y_train)

# Check the predictions made to the test set.
predictions = linear_regression_model.predict(X_test)

# Error made by the model
error_in_model = mean_squared_error(predictions, y_test, squared=False)

# Metric for the model: R-squared
model_performance = linear_regression_model.score(X_test, y_test)

print(f"""
RMSE (Error made by the model)
------------------------------
The model has an average error of {error_in_model}$

R-squared (Metric of the model)
-------------------------------
The model's performance is {round(model_performance, 6)}
""")
