Now that we can make predictions using a classifier. How can you know if a model is making correct predictions?
We can evaluate its performance.

In classification accuracy is a commonly-used metric.

$$accuracy = \frac {correct\ predictions} {total\ observation} $$

We could compute accuracy on the data used to fit the classifier. However, as this data was used to train the model, performance will not be indicative of how well it can generalize to unseen data.

It is common to split the data into a training set and a test set.

We fit the classifier using the training set, then we can calculate the model's accuracy against the test set labels.

> from sklearn.model_selection import train_test_split
> X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=21, stratify=y)

It is best practice to ensure our split reflects the proportion of labels in our data. For example: if churn occurs in 10% of observations, we want 10% of labels in our training and test sets to represent churn.

> knn = KNeighborsClassifier(n_neighbors=6)
> knn.fit(X_train, y_train)
> print(knn.score(X_test, y_test))
> 0.88

https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=7

# Model complexity
Lets discuss how to interpret k. Recall that we discussed decision boundaries, which are thresholds for determining what label a model assigns to an observation.

As k increases, the decision boundary is less affected by individual observations, reflecting a simpler model. 

![[Pasted image 20230925081651.png]]

Simpler models are less able to detect relationships in the dataset, which is known as underfitting. 
In contrast complex models can be sensitive to noise in the training data, rather than reflecting general trends, this is known as overfitting.

## Model complexity curve
Se can also interpret K using a model complexity curve, with a KNN model, we can calculate accuracy on the training and test sets using incremental K values, and plot the results.

> train_accuracies = {}

```python 
train_accuracies = {}
test_accuracies = {}

neighbors = np.arrange(1, 26)


```