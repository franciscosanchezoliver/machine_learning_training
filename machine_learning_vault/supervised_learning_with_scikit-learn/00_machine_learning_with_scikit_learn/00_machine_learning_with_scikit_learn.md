<h1>Machine learning with scikit-learn</h1>

In Machine learning computers learn to make decisions from data without 
being explicitly programmed.

Examples: 
- Predicting if an email is spam or not given its content and sender. 
- Learn to cluster books to one exiting cluster based on the words they contain, then assigning any new book to one of the existing clusters.


<h2>Unsupervised Learning</h2>
Unsupervised learning is the process of __uncovering hidden patterns 
and structures from unlabeled data__.

<h3>Example</h3>
A business may wish to __group its customers into distinct categories based 
on their purchasing behavior without knowing in advance what these 
categories are__. This is known as clustering, one branch of unsupervised learning.

![cluster analysis for customer churn](imgs/cluster_analysis_for_customer_churn.png)


<h2>Supervised Learning</h2>
Supervised learning is a __type of machine learning where the values to 
be predicted are already known__, and a model is built with the aim of 
accurately predicting values of previously unseen data.

Supervised learning uses features to predict the values of a 
target variable. Such as predicting a basketball player's position based 
on their points per game.

![features and target for basketball](imgs/features_and_target_for_basket.png)

We'll focus on supervised learning for now.

## Types of supervised learning
There are __2 types of supervised learning__:
- __Classification__: used to __predict the label (or category) of an observation__.

	Example: predict whether a bank transaction is fraudulent or not. As there 
    are two outcomes here fraudulent/not fraudulent this is known as 
  	binary classification. <br><br>

- __Regression__: used to predict continuous values.<br>
    Example: a model can use features such as number of bedrooms, ad the size 
    a property, to predict the price of the property.


## Before using supervised learning
1. There are some requirements to satisfy before performing supervised learning. 
   - Our data must __not have missing values__. 
   - Must be in a __numeric format__.

2. Perform some __exploratory analysis first__ to ensure data is in the correct format.
   - Descriptive statistics
   - Data visualizations

## Binary classification

We saw that there are two types of supervised learning: classification and 
regression. Recall that __binary classification__ is used to __predict a 
target variable that has only two labels__, typically represented numerically 
with a zero or a one.

### Exercise:
Given the following dataset:

![churn data example](imgs/churn_dataset_example.png)


Which column could be the target variable for binary classification?

- [ ] customer_service_calls
- [ ] total_night_charge
-  [X] churn
	has values of 0 or 1, so it can be predicted using a binary classification model.
- [ ] account_length

## The classification challenge
Let's discuss how we can build a classification model, or classifier, to predict 
the labels of unseen data.

Steps to build the model:
- Build a classifier which learns from the labeled data we pass to it.
- Then we pass unlabeled data as input, and have it predict labels for this unseen data.

As the classifier learns from the labeled data, we call this the training data.


## K-Nearest Neighbors (KNN)

We'll use an algorithm called __K-Nearest Neighbors, which is popular for classification problems__. 
The idea of KNN is to __predict the label of any data point by looking at the k (example 3) closest 
labeled data__. And guessing them to vote on what label the unlabeled observation should have.

__KNN uses majority voting__, which makes predictions based on what label the majority of nearest 
neighbors have. 

Using the following scatter plot. 

![knn point not classified](imgs/knn_point_not_classified.png)


How do we classify the black point?
If k = 3, then we would classify it as a red point, this is because 2 of the 3 
closest observations are red

![unclassified point and neighbors](imgs/unclassified_point_and_neighbors.png)

if k = 5, we would classify it as blue.

![point classified as blue with k equal 5](imgs/unclassified_point_with_k_5.png)

To build some intuition for KNN, let's look at this scatter plot displaying 
total evening charge against total charge for customers of a telecom company. 

The observations in blue for customers who have churned, and red for 
customers that have not churned. 

![knn evening charge vs total charge](imgs/knn_evening_charge_vs_total_day_charge.png)

Here, we have visualized the results of a KNN algorithm where the number 
of neighbors is 15.

![Result of applying knn equals 15](imgs/result_of_applying_knn_15.png)

KNN creates a decision boundary to predict if customers will churn. So, any 
customer in the gray area are predicted to churn, and those in the area 
with a red background are predicted to not churn.

This boundary would be used to make predictions on unseen data. 

### Exercise
[Exercise](https://github.com/spuzi/machine_learning_training/blob/main/knn/00_knn_fit.py)

```python
"""
k-Nearest Neighbors: Fit
------------------------
In this exercise, you will build a simple classification model using the churn_df dataset.
The features to use will be "account_length" and "customer_service_calls". The target, "churn",
needs to be a single column with the same number of observations as the feature data.

You will convert the features and the target variable into NumPy arrays, create
an instance of a KNN classifier, and then fit it to the data.


To do:
- Extract the features array, for this case "account_length" and "customer_service_calls".
- Extract the target array, the column "churn".
- Create the KNN model using 6 neighbors.
- Fit the classifier.

"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

churn_data = pd.read_csv('churn_data_modified.csv')

X = churn_data[['account_length', 'customer_service_calls']].values
y = churn_data["churn"].values

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X, y)

print(churn_data)

"""
Now you have fit a KNN classifier, you can use it to predict the label of new data points. All available 
data was used for training, however, fortunately, there are new observations available (X_new). 

You will use your classifier to predict the labels of a set of new data points
"""
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

print("New data point")
print(X_new)

predictions = knn.predict(X_new)
print(f"Predictions done for new data point: {predictions}")
```