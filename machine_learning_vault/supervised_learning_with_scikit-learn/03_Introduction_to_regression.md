In regression tasks, the target variable typically has continuous values, such a country's GDP, or the price of a house.

For example, we could use a dataset containing health data to predict blood glucose levels.

```python
import pandas as pd

diabetes_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head())
```

![[Pasted image 20230929073619.png]]

```python
X = diabetes_df.drop('diabetes', axis=1).values
y = diabetes_df['diabetes'].values
```

Let's try to predict blood glucose levels from a single feature: body mass index.
```python
X_bmi = X[:, 3]
X_bmi = X_bmi.reshape(-1, 1)

import matplotlib.pyplot as plt

plt.scatter(X_bmi, y)

plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")

plt.show()
```

![[Pasted image 20230929074418.png]]o

We can see that generally, as body mass increases, blood glucose levels also tend to increase.


## Fitting a regression model
Lets fit a regression model to our data. We're going to use a model called "linear regression", which fits a straight line to our data.


```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# Plot the real data
plt.scatter(X_bmi, y)

# Plot the predicted data (it's going to be a line)
plt.plot(X_bmi, predictions)


plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")

plt.show()
```

![[Pasted image 20230929075641.png]]

The relation between blood glucose and body mass index appear to have a moderate positive correlation.

## Exercise:
- [Exercise](https://github.com/spuzi/machine_learning_training/blob/main/regression/00_creating_features.py)

## Basics of linear regression

We want to fit a line to the data, and in 2 dimensions this takes the form of:
$$ y = ax + b$$
Using a single feature is known as 'simple linear regression', where _y_ is the target, _x_ is the feature and  _a_ and _b_ are the model parameters that we want to learn.

_a_ and _b_ are also called the 'model coefficients', or a (slope) and b (intercept).

To choose values for _a_ and _b_ we define an error function and then choose the parameters that minimize this function (line that minimize the error function).

Error functions are also called _loss_ or _cost functions_. 

We want the line to be as close to the observations as possible. Therefore, we want to minimize the vertical distance between the fit and the data:

![[Pasted image 20230930085307.png]]

The distance between each point and the line is called _Residual_. 

We could try to minimize the sum of residuals but then each positive residual would cancel out each negative residual. 
![[Pasted image 20230930090027.png]]

To avoid this we squared the residuals. By adding all the squared residuals, we calculate the sum of squares, or RSS.
https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-6320c92e-31c3-48fb-9382-6a9169125722?ex=5

$$ RSS = \sum_{i=1}^{n} (y_i - \hat{y_i})^2  $$
This type of linear regression is called _Ordinary Least Squares (OLS)_ while we trying to minimize the RSS. 

When we have 2 features and one target the line will have the following formula:
$$ y = a_1  x_1 + a_2 x_2 + b $$

So, to fit a linear regression model we need to specify 3 variables: 
$$a1, a2, b$$

When adding more features, it is known as multiple linear regression.

Fitting a multiple linear regression model means specifying _n_ coefficients for a's (one per feature) and one _b_.








