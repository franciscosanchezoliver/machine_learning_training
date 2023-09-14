We'll learn how to use XGBoost for regression. Regression problems involve predicting continuous, or real, values. 

For example, if you are attempting to predict the height in centimeters of a person given some of their physical attributes at birth, then you are solving a regression problem. 

Evaluating the quality of a regression model involves using a different set of metrics than those that we described for use in classification problems. In most cases we use root mean squared error (RMSE) or the mean absolute error (MAE) to evaluate the quality of a regression model. 

## Computing RMSE

- Taking the difference between the actual and the predicted values for what you are trying to predict. 

 | Actual | Predicted |
 | ------ | --------- |
 | 10     | 20        |
 | 3      | 8         |
 | 6      | 1         |

- Squaring those differences

 | Actual | Predicted | Error |  Squared Error   |
 | ------ | --------- | ----- | --- |
 | 10     | 20        | -10   | 100    | 
 | 3      | 8         | -5    |  25   |
 | 6      | 1         | 5     |  25   |


- Computing their mean of the squared errors:
$$ (100 + 25 + 25)/3 = 50 $$

-  Taking the squared root of the mean squared errors.
$$ \sqrt{50} = 7.07$$


Squaring the differences allow us to treat negative and positive differences equally, but tends to punish larger differences between predicted and actual values much more than smaller ones.

## Computing MAE (Mean Absolute Error)

MAE simply sums the absolute differences between predicted and actual values across all of the examples.


 | Actual | Predicted | Error |
 | ------ | --------- | ----- |
 | 10     | 20        | -10   |
 | 3      | 8         | -5    |
 | 6      | 1         | 5     |


https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/regression-with-xgboost?ex=1