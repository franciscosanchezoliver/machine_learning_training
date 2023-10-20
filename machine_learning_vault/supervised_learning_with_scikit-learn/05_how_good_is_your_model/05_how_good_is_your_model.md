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


Given a binary classifier, such as our fraudulent transaction example. We can create a 2 by 2 matrix that summarizes performance called a _'confusion matrix'_.


|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     False Positive    |
| Actual: Fraudulent | False Negative        |     True Positive     |

Given any model, we can fill in the confusion matrix according to its predictions.

The true positive are the number of fraudulent transactions correctly labeled.

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     False Positive    |
| Actual: Fraudulent | False Negative        |     __True Positive__     |

The true negatives are the number of legitimate transactions correctly labeled.

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | **True Negative**         |     False Positive    |
| Actual: Fraudulent | False Negative        |     True Positive     |


The false negatives are the number of legitimate transactions incorrectly labeled

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     False Positive    |
| Actual: Fraudulent | **False Negative**        |     True Positive     |


The false positives are the number of transactions incorrectly labeled as fraudulent

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     **False Positive**    |
| Actual: Fraudulent | False Negative        |     True Positive     |

Usually, the class of interest is called the '*Positive Class*'.

As, we aim to detect fraud, the positive class is an illegitimate transaction.

## Why is the confusion matrix so important

### Accuracy

Firstly, we can retrieve 'accuracy', which is the sum of true predictions divided by the total sum of the matrix:

$$ accuracy = \frac{t_p + t_n}{t_p + t_n + f_p + f_n} $$



### Predictions

Secondly, there are other important metrics we can calculate from the confusion matrix. 

*Precision*, its the number of true positive divided by the sum of all positive predictions.

$$ precision = \frac {t_p} {t_p + f_p} $$

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     **False Positive**    |
| Actual: Fraudulent | False Negative        |     **True Positive**     |


High precision means having a lower false positive rate.

In our classifier, this would translates to fewer legitimate transactions being classified as fraudulent.

## Recall

Is the number of true positives divided by the sum of true positives and false negatives. This is also called as '*sensitivity*'.

$$ recall = \frac {t_p} {t_p + f_n}  $$

High recall reflects a lower false negative rate.

For our classifier, it means predicting most fraudulent transactions correctly.


## F1 Score

The F1-score is the harmonic mean of precision and recall.

$$ F1\ Score =  2 * \frac {precision * recall} {precision + recall} $$

This metric gives equal weight to precision and recall. Therefore, it factors in both the number of errors made by the model and the type of errors.


F1 score favors models with similar precision and recall, and it is useful metric if we seek a model which performs reasonably well across both metrics.


## Code to calculate the confusion matrix

Using the churn dataset, to compute the confusion matrix, along with the metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix

# We instantiate a classifier
knn = KNeighborsClassifier(n_neighbors=7)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
												   test_size=0.3,
												   random_state=42)

# Fit the classifier with the training data
knn.fit(X_train, y_train)

# Predict with the test set
y_pred = knn.predict(X_test)

# We pass the predicted labels, and the actual test set labels
# to calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# We can see 1106 True Negatives.
# [[1106    11]
#  [ 183    34]]

# Passing the same arguments as in the confusion matrix to the
# method "classification_report" outputs all the relevant metrics

print(classification_report(y_test, y_pred))

```





