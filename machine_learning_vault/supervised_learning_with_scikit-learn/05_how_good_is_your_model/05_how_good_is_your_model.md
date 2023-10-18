https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model-3?ex=1

thinking back to classification problems recall that we can use accuracy, the fraction of correctly classified labels, to measure model performance. 

$$accuracy = \frac {correct\ predictions} {total\ observation} $$

However, accuracy is not always an useful metrics. 

Consider a model for predicting whether a bank transaction is fraudulent, where only 1% of transactions are actually fraudulent.
- 99% legitimate transactions
- 1% fraudulent transactions

We could build a model that classifies every transaction as legitimate, this model would have an accuracy of 99%. However it does a terrible job of actually predicting fraud, so it fails at its original purpose.

The situation where one __class is more frequent is called _'class imbalance'____.

Here, the class of fraudulent transactions contains more instances that the class of legitimate transactions.

This is a common situation in practice, and require a different approach to assessing the model's performance.


Given a binary classifier, such as our fraudulent transaction example. 



