We know now how to split into train and test sets, and computing the model's performance on our test set.

There is a potential pitfall for this process.

If we're computing R-squared on our test set, the R-squared returned is dependent on the way we split up the data.

The data points in the test set may have some peculiarities that means that the R-squared computed on it is not representative of the model's ability to generalize to unseen data.

To combat this dependence, we use a technique called _cross-validation_.

We begin by splitting our data into groups or folds. Then we set aside the first fold as the test set, fit our model on the remaining folds, and predict on our test set.

![[Pasted image 20231002082320.png]]

https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-6320c92e-31c3-48fb-9382-6a9169125722?ex=8