Regularized regression is a technique used to avoid overfitting.

Recall that fitting a linear regression minimizes a loss function to choose a coefficient _a_ (for each feature) and the intercept _b_.

__If we allow these coefficients to be very large, we can get overfitting__. Therefore, it is common practice to alter the loss function so that it __penalizes large coefficients__. This is called _regularization_.


## Ridge Regression

The first type of regularized regression that we'll look at is called ridge. With ridge, we use the Ordinary Least Squares (OLS) loss function:

---------------
![[Pasted image 20230930090027.png]]

To avoid this we squared the residuals. By adding all the squared residuals, we calculate the sum of squares, or RSS.

$$ RSS = \sum_{i=1}^{n} (y_i - \hat{y_i})^2  $$
This type of linear regression is called _Ordinary Least Squares (OLS)_ while we trying to minimize the RSS. 

--------------------------

So we can calculate the Ridge loss function with the sum of the normal OLS loss function and the _normalization_ that is calculated multiplying a alpha to the sum of the squared values of each coefficient.
Plus the squared value of each coefficient, multiplied by a constant, alpha. 
$$Ridge\ loss\ function = OLS\ loss\ function + \alpha * \sum_{i=1}^{n} a_i^2 $$
So, when minimizing the loss function, _models are penalized for coefficients with large positive or negative values_.

When using ridge _we need to choose the alpha value_ in order to fit and predict.

Essentially, we can _select the alpha for which our model performs best_.

Alpha controls model complexity, __when alpha = 0__ then there is __no normalization__, so its just a OLS function and __this can lead to overfitting__.

A large alpha means that large coefficients are significantly penalized, which can lead to underfitting.

```python
from sklearn.linear_model import Ridge

scores = []

# Let's try different values of alpha to see how 
# this affect the outcome of the model.
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:

	# Create a Linear regression with Ridge regularization
	ridge = Ridge(alpha = alpha)

	# Fit the model with the training data



```


https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-6320c92e-31c3-48fb-9382-6a9169125722?ex=11




