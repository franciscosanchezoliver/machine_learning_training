Now that we can make predictions using a classifier. How can you know if a model is making correct predictions?
We can evaluate its performance.

In classification accuracy is a commonly-used metric.

$$accuracy = \frac {correct\ predictions} {total\ observation} $$

We could compute accuracy on the data used to fit the classifier. However, as this data was used to train the model, performance will not be indicative of how well it can generalize to unseen data.

It is common to split the data into a training set and a test set.

We fit the classifier using the training set, then we can calculate the model's accuracy against the test set labels.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,  
													test_size=0.3,
													random_state=21,
													stratify=y)
```

It is best practice to __ensure our split reflects the proportion of labels in our data__. For example: if churn occurs in 10% of observations, we want 10% of labels in our training and test sets to represent churn.

```python
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
0.88
```


https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=7

# Model complexity
Lets discuss how to interpret k. Recall that we discussed decision boundaries, which are thresholds for determining what label a model assigns to an observation.

As k increases, the decision boundary is less affected by individual observations, reflecting a simpler model. 

![[Pasted image 20230925081651.png]]

Simpler models are less able to detect relationships in the dataset, which is known as underfitting. 
In contrast complex models can be sensitive to noise in the training data, rather than reflecting general trends, this is known as overfitting.

## Model complexity curve
Se can also interpret K using a model complexity curve, with a KNN model, we can calculate accuracy on the training and test sets using incremental K values, and plot the results.

```python 
train_accuracies = {}
test_accuracies = {}

neighbors = np.arrange(1, 26)

for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors=neighbor)
	knn.fit(X_train, y_train)

	# Store the accuracies obtain on the train and test set.
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
```

Now we can plot the different scores obtain using a different value of K. 
```python
plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of Neighbors")

plt.plot(neighbors, 
		 train_accuracies.values(), 
		 label='Training accuracies')

plt.plot(neighbors, 
		 test_accuracies.values(), 
		 label='Test accuracies')

plt.legend()
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")

plt.show()
```

Here's the result:

![[Pasted image 20230927072859.png]]

As k increases beyond 15 we see overfitting where performance plateaus on both test and training sets, as indicated in this plot.

The peak test accuracy actually occurs at around 13 neighbors.
![[Pasted image 20230927073432.png]]

## Exercises

- Split data into train and test
- Use a KNN classifier

1. [Exercise](https://github.com/spuzi/machine_learning_training/blob/main/knn/01_knearest_neighbors_train_test_accuracy_comparison.py)
	- Calculate the score with the test set.

2. [Exercise](https://github.com/spuzi/machine_learning_training/blob/main/knn/02_knn_overfitting_underfitting.py)
	- Model Complexity Curve


