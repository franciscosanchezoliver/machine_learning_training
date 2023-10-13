"""
Linear Regression with one feature
----------------------------------

You will work with a dataset called with marketing sales data, which contains information
on advertising campaign expenditure across different media types, and the number of
dollars generated in sales for the respective campaign.

Example of the dataset:

     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83

We will try to predict the amount of money earn just by seeing the amount of money invested in radio.

To do:
    - Use a linear regression model to predict the money earn using the
      money invested in radio
    - Visualize the line created by the linear regression model.
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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

# Show the money invested in radio, the real money obtained and the predicted one.
for i in range(10):
    investment_done_in_radio = round(X[i][0], 2)

    real_benefit_obtain = round(y[i])
    predicted_one = round(predictions_made[i], 2)

    print(
        f"{investment_done_in_radio} => {real_benefit_obtain}, {predicted_one} | {round(abs(real_benefit_obtain - predicted_one))}")

# Plot the real values
plt.scatter(X, y)

# Draw a line with the predictions made
plt.plot(X, predictions_made, color='red')

# Set a name for the axis
plt.title("Radio Expenditure and Sales")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

plt.show()
