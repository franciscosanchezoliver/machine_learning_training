"""
Decision trees
--------------
Your task in this exercise is to make a simple decision tree using 
scikit-learn's DecisionTreeClassifier on the breast cancer dataset that comes 
preloaded with scikit-learn.

This dataset contains numeric measurements of various dimensions of individual 
tumors (such as perimeter and texture) from breast biopsies and a single outcome
value (the tumor is either malignant, or benign).

To do:
- Split the dataset into train and test.
    - Create training and test sets such that 20%
      of the data is used for testing. Use a random_state of 123.

- Train a Decision tree classifier. You'll specify a parameter called max_depth. 
Many other parameters can be modified within this model, and you can check all 
of them out here.
    - max_depth of 4. This parameter specifies the maximum
      number of successive split points you can have before reaching
      a leaf node.

- Fit the classifier to the training set and predict the labels of
  the test set.
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from sklearn.
cancer_dataset = load_breast_cancer()

# Create a pandas DataFrame with the loaded dataset
df_dataset = pd.DataFrame(cancer_dataset.data,
                          columns=cancer_dataset.feature_names)

# Select the features and target variable
X = df_dataset
y = cancer_dataset['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123
)

# Create a decision tree classifier object
decision_tree_classifier = DecisionTreeClassifier(max_depth=4)

# Fit it to the training set.
decision_tree_classifier.fit(X_train, y_train)

# Predict with the test set labels
predictions = decision_tree_classifier.predict(X_test)

# Calculate accuracy as metric.
times_correctly_predicted = np.sum(predictions == y_test)
size_of_test_set = y_test.shape[0]
accuracy = times_correctly_predicted / size_of_test_set
print(f"Accuracy: {accuracy}")
