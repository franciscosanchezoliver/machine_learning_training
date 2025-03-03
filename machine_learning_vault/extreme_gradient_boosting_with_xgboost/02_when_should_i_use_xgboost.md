#xgboost #supervised
# When should I use XGBoost
- Consider using XGBoost for any supervised machine learning task that fits the following criteria:
	- **Large number of training examples**. 
		- At least few feature and at least 1000 examples
		- In general, as long as **the number of features in your training set is smaller than the number of examples** you have, you should be fine. 
	- XGBoost tends to do well **when you have mixture of categorical and numerical features, or when you have just numeric features.**

- When you should not use XGBoost? The most important kind of problems where XGBoost is a sub optimal choice involve either those that have found success using other state of the art algorithms or those that suffer from dataset size issues.
	- Specifically, **XGBoost is not ideally suited for image recognition, computer vision, or natural language processing** and understanding problems. As those kind of problems can be much better tackled using deep learning approaches. 
	- In term of dataset size problems, XGBoost **is not suitable when you have very small training sets, like when you have fewer than a 100 training samples or when the number of training samples is significantly smaller than the number of features being used for training. **

# Exercise: Using XGBoost
XGBoost is a powerful library that scales very well to many samples and works for a variety of supervised learning problems. But you shouldn't always pick it as your default machine learning library when starting a new project, since there are some situations in which it is not the best option. In this exercise, your job is to consider the below examples and select the one which would be the best use of XGBoost.

#### Possible Answers

- [ ] Visualizing the similarity between stocks by comparing the time series of their historical prices relative to each other.
	- This is an example of a clustering problem, there are no labels to learn from here.
 
- [ ]  Predicting whether a person will develop cancer using genetic data with millions of genes, 23 examples of genomes of people that didn't develop cancer, 3 genomes of people who wound up getting cancer.
	- This would not be an ideal use of XGBoost as there are many more features than there are examples.

- [ ]  Clustering documents into topics based on the terms used in them.
	- This is an example of a clustering problem. There are no targets and so you cannot use supervised learning.
 
-  [X] Predicting the likelihood that a given user will click an ad from a very large clickstream log with millions of users and their web interactions.


