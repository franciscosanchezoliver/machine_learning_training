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
from sklearn.model_selection import train_test_split
import pandas as pd

sales_df = pd.read_csv('advertising_and_sales_clean.csv')

# We'll select the features that are not categorical.
features = ['tv', 'radio', 'social_media']
target = ['sales']

# Select the features and target variables
X = sales_df[features]
y = sales_df[target]

# Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Let's see the score obtain when varying the parameter alpha for the
# ridge regularization
# Create a list for the alpha values from 0.1 to 1000
alpha_values = [10 ** i for i in range(-1, 5)]

print(
    """
    Alpha controls model complexity, when alpha = 0 then there is no normalization, so its 
    just a OLS function and this can lead to overfitting.
    
    A large alpha means that large coefficients are significantly penalized, which 
    can lead to underfitting.
    """
)

for each_alpha in alpha_values:
    # Create a linear regression model with a ridge regularization
    ridge = Ridge(alpha=each_alpha)

    # Train the model using the training set
    ridge.fit(X_train, y_train)

    # Get the predictions made
    sales_predicted = ridge.predict(X_test)

    # Get the score achieved with the test set
    score_achieved = ridge.score(X_test, y_test)

    print(f"""
    Score => {score_achieved} | Alpha => {each_alpha} 
        Predicted sample: {
    [round(prediction[0], 2) for prediction in sales_predicted[:5]]
    }
    """)

print("""
    The scores don't appear to change much as alpha increases, which is indicative 
    of how well the features explain the variance in the target, even by heavily 
    penalizing large coefficients, underfitting does not occur!
""")
