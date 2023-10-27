"""
Assessing a diabetes prediction classifier
------------------------------------------
In this exercise you'll work with a diabetes dataset.

The goal is to predict whether each individual is likely to have diabetes based on
the features body mass index (BMI) and age (in years).

Therefore, it is a binary classification problem. A target value of 0 indicates
that the individual does not have diabetes, while a value of 1 indicates that
the individual does have diabetes.


To do:
- Split the data into train and test.
- User a KNN classifier to predict who has diabetes.
- Create a confusion matrix and a classification report

"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd

# Read the dataset
diabetes_df = pd.read_csv('diabetes_clean.csv')

# Select the features and the target.
# We will use Body Mass Index (BMI) and the age as the features to predict whether
# the person has diabetes.
features = ["bmi", "age"]
target = ["diabetes"]
X = diabetes_df[features]
y = diabetes_df[target]

# Split into train and test sets
# We'll use the 30% of the dataset for training.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=30,
                                                    stratify=y,
                                                    random_state=42)

# Create a classification model.
# For this case we'll use the KNN classifier model.
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model with the training set
knn.fit(X_train, y_train.values.ravel())

# Predict whether the individuals have diabetes in the test set
y_pred = knn.predict(X_test)

# Generate the confusion matrix and the classification report
the_confusion_matrix = confusion_matrix(y_test, y_pred)
the_classification_report = classification_report(y_test, y_pred)

# True Negative: the individual doesn't have diabetes and the model got it right
true_negative = the_confusion_matrix[0][0]

# True Positive: the individual has diabetes and the model got it right
true_positive = the_confusion_matrix[1][1]

# False Negative: the model said that the individual doesn't have diabetes, but he has.
# This is the worst case
false_negative = the_confusion_matrix[1][0]

# False Positive: the model said that the individual has diabetes, but he hasn't
false_positive = the_confusion_matrix[0][1]

print(f"""
                        -----------------------------------
                       |              Predicted            |
                        -----------------------------------
                       |  Not Diabetes   |     Diabetes    |
 ----------------------------------------------------------
|      | Not Diabetes  | True Negative   |  False Positive |
|Real  -----------------------------------------------------
|      | Diabetes      | False Negative  |  True Positive  |
 -----------------------------------------------------------
 
{true_negative}   |  {false_positive} |
{false_negative}  |  {true_positive}  |
 
True Negatives ({true_negative}): The individual doesn't have diabetes and the model got it right.
True Positives ({true_positive}): The individual has diabetes and the model got it right.


!False Negative ({false_negative}): The model said that the individual doesn't have diabetes but 
                                    the individual does have diabetes. This is the worst case.
False Positive ({false_positive}): The model said that the individual has diabetes but the individual 
                                   hasn't got diabetes.
                                   
We can check that the built in function confusion_matrix in sklearn gives the same
output for the confusion matrix that we've just done.
{the_confusion_matrix}

""")

# Now let's calculate the most important metrics that we can get from the confusion matrix

# Accuracy
# accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

# Precision
# Number of true positives divided by the sum of all positives predictions
# precision = Tp / (Tp + Fp)
precision = true_positive / (true_positive + false_positive)

# Recall
# Number of true positives divided by the sum of true positives and false negatives.
#
# recall = Tp / (Tp + Fn)
#
# For this problem this is the most important metrics, as we could predict that someone
# hasn't got diabetes when he has.
recall = true_positive / (true_positive + false_negative)

# F1-Score
# It's the harmonic mean between precision and recall.
# This metric gives equal weight to precision and recall. Therefore, it factors
# in both the number of errors made by the model and the type of error.
# For this problem I think that the recall its more important that the precision.
# But let's calculate the F1-Score anyway.
# F1-Score = 2 * (precision * recall) / (precision + recall)
f1_score = 2 * (precision * recall) / (precision + recall)

# We can retrieve all this information with the function "classification_report" from sklearn.


print(f"""
Metrics from confusion matrix
-----------------------------
Accuracy: {accuracy}
Precision {precision}
!Recall: {recall}
F1-Score: {f1_score}

We can get the same results by calling the function from scikit learn:
{the_classification_report}
""")
