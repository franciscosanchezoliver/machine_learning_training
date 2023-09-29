"""
Creating features
-----------------
You will work with a dataset called "sales_df", which contains information on advertising campaign
expenditure across different media types, and the number of dollars generated in sales for the
respective campaign.

An example of the dataset:

     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83

You will use the advertising expenditure as features to predict sales values, initially working
with the "radio" column. However, before you make any predictions you will need to create
the feature and target arrays, reshaping them to the correct format for scikit-learn.
"""

from sklearn.linear_model import LinearRegression
import pandas as pd

sales_df = pd.read_csv('./advertising_and_sales_clean.csv')

X = sales_df['radio'].values
y = sales_df['sales'].values

# As we have select only one feature we need to reshape
X = X.reshape(-1, 1)

# Create a Linear Regression model
reg = LinearRegression()

# Fit it to the data
reg.fit(X, y)

# Predict with our model
predictions_made = reg.predict(X)

# Compare the predictions made with the actual values

print("Showing the first 10 predictions made")
print("Investment in Radio => Real Benefit, Predicted  | diff")
for i in range(10):
    investment_done_in_radio = round(X[i][0], 2)

    real_benefit_obtain = round(y[i])
    predicted_one = round(predictions_made[i], 2)

    print(
        f"{investment_done_in_radio} => {real_benefit_obtain}, {predicted_one} | {round(abs(real_benefit_obtain - predicted_one))}")
