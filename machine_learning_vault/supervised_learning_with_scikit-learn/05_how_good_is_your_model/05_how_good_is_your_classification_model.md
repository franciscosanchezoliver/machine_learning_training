#accuracy #precision #recall #confusion_matrix #classification
# How good is your classification model?


We've used accuracy in classification problems to measure the fraction 
of correctly classified labels. 

$$accuracy = \frac {correct\ predictions} {total\ observation} $$

However, accuracy isn't always a useful metrics. 

Consider a model for predicting whether a bank transaction is fraudulent 
where only 1% of transactions are actually fraudulent.

- 99% legitimate transactions
- 1% fraudulent transactions

## Class imbalance

We could build a model that classifies every transaction as legitimate, this 
model would have an accuracy of 99%. However, it does a terrible job of 
actually predicting fraud, so it fails at its original purpose.

The situation where one __class is more frequent is called _'class imbalance'____.

Here, the class of fraudulent transactions contains more instances that the 
class of legitimate transactions.

This is a common situation in practice, and require a different approach to 
assessing the model's performance.

## Confusion matrix

Given a binary classifier, such as our fraudulent transaction example. We can 
create a 2 by 2 matrix that summarizes performance called a _'confusion matrix'_.

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     False Positive    |
| Actual: Fraudulent | False Negative        |     True Positive     |

Given any model, we can fill the confusion matrix according to its predictions.

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

It's so important because we can calculate several metrics from it.

### Accuracy

The sum of true predictions divided by the total sum of the matrix:

$$ accuracy = \frac{t_p + t_n}{t_p + t_n + f_p + f_n} $$


### Precision

It's the number of **true positive divided by the sum of all positive predictions**.

$$ precision = \frac {t_p} {t_p + f_p} $$

|                    | Predicted: Legitimate | Predicted: Fraudulent |
|--------------------|-----------------------|-----------------------|
| Actual: Legitimate | True Negative         |     **False Positive**    |
| Actual: Fraudulent | False Negative        |     **True Positive**     |


**High precision → lower false positive rate**.

In our classifier, this would translate to fewer legitimate transactions being 
classified as fraudulent.

## Recall

Is the number of **true positives divided by the sum of true positives and 
false negatives**. 

This metric is also called as '*sensitivity*'.

$$ recall = \frac {t_p} {t_p + f_n}  $$

**High recall → lower false negative rate.**

For our classifier, it means predicting most fraudulent transactions correctly.


## F1 Score

**It's the harmonic mean of precision and recall.**

$$ F1\ Score =  2 * \frac {precision * recall} {precision + recall} $$

This metric **gives equal weight to precision and recall**. Therefore, **it factors 
in both the number of errors made by the model and the type of errors.**

**F1 score favors models with similar precision and recall**, and it is useful 
metric if we seek a model which performs reasonably well across both metrics.


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

#                  precision   recall    f1-score  support
#            0          0.86     0.99        0.92     1117
#            1          0.76     0.16        0.26      217
#
#
#     accuracy                               0.85     1334
#    macro avg          0.81     0.57        0.59     1334
# weighted avg          0.84     0.85        0.81     1334
```

## Exercise: Deciding on a primary metric

As you have seen, several metrics can be useful to evaluate the performance 
of classification models, including: accuracy, precision, recall, and F1-score.

In this exercise, you will be provided with three different classification problems, and 
your task is to **select the problem where precision is best suited as the primary metric**.

-  [ ] A model predicting the presence of cancer as the positive class.
		**This model should minimize the number of false negatives, so recall is a 
	 	more appropriate metric.**
		**High recall → lower false negative rate.**
		$$ recall = \frac {t_p} {t_p + f_n}  $$

  - [ ] A classifier predicting the positive class of a computer program containing malware.
      To avoid installing malware, **false negatives should be minimized, 
      hence recall or F1-score are better metrics for this model**.
 
      **High recall → lower false negative rate.**
  		$$ recall = \frac {t_p} {t_p + f_n}  $$

-  [X] A model predicting if a customer is a high-value lead for a sales team with limited capacity.
	Correct! With limited capacity, the sales team needs the model to return the highest proportion of true positives compared to all predicted positives, thus minimizing wasted effort.

	**High precision → lower false positive rate**.
$$ precision = \frac {t_p} {t_p + f_p} $$

## Exercise: assessing a diabetes prediction classifier
[Classifying whether an individual has diabetes](https://github.com/franciscosanchezoliver/machine_learning_training/blob/main/knn/05_assesing_a_classifier/assesing_a_classifier.py)
