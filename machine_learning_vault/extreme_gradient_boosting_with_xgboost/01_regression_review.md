We'll learn how to use XGBoost for regression. Regression problems involve predicting continuous, or real, values. 

For example, if you are attempting to predict the height in centimeters of a person given some of their physical attributes at birth, then you are solving a regression problem. 

Evaluating the quality of a regression model involves using a different set of metrics than those that we described for use in classification problems. In most cases we use root mean squared error (RMSE) or the mean absolute error (MAE) to evaluate the quality of a regression model. 

RMSE is computed by 
	- Taking the difference between the actual and the predicted values for what you are trying to predict. 
	- Squaring those differences
	- Computing their mean

https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/regression-with-xgboost?ex=1