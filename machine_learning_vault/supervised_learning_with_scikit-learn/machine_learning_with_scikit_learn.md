# Machine learning with scikit-learn

https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=1
In Machine learning computers learn to make decisions from data without being explicitly programmed.
Examples: 
	- Predicting if an email is spam or not given its content and sender. 
	- Learn to cluster books to one exiting cluster based on the words they contain, then assigning any new book to one of the existing clusters.


## Unsupervised Learning
Unsupervised learning is the process of uncovering hidden patterns and structures from unlabeled data.
Example: 
A business may wish to group its customers into distinct categories based on their purchasing behavior without knowing in advance what these categories are. This is known as clustering, one branch of unsupervised learning.
![[Pasted image 20230913073612.png]]

## Supervised Learning
Supervised learning is a type of machine learning where the values to be predicted are already known, and a model is built with the aim of accurately predicting values of previously unseen data.

Supervised learning uses features to predict the values of a target variable. Such as predicting a basketball player's position based on their points per game.
![[Pasted image 20230913074237.png]]

We'll focus on supervised learning for now.

## Types of supervised learning
There are 2 types of supervised learning:
- Classification: used to predict the label (or category) of an observation.
	Example: predict whether a bank transaction is fraudulent or not. As there are two outcomes here fraudulent/not fraudulent this is known as binary classification.
 
- Regression: used to predict continuous values. 
	Example: a model can use features such as number of bedrooms, ad the size of a property, to predict the price of the property.


## Before you use supervised learning
There are some requirements to satisfy before performing supervised learning. 
- Our data must not have missing values. 
- Must be in a numeric format.

Perform some exploratory analysis first to ensure data is in the correct format.